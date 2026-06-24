# Copyright (c) Meta Platforms, Inc. and affiliates.

# References:
#  https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_sd3.py
#  https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py

import copy
import inspect
import math
import sys

import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    PNDMScheduler,
)
from torch.nn import functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress is optional at runtime.
    tqdm = None


SCHEDULER_MAPPING = {
    "DDIMScheduler": DDIMScheduler,
    "DDPMScheduler": DDPMScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "PNDMScheduler": PNDMScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
}

FLOW_VELOCITY_NOISE_MINUS_X0 = "noise_minus_x0"
FLOW_VELOCITY_X0_MINUS_NOISE = "x0_minus_noise"

_FLOW_VELOCITY_ALIASES = {
    FLOW_VELOCITY_NOISE_MINUS_X0: FLOW_VELOCITY_NOISE_MINUS_X0,
    "partgen": FLOW_VELOCITY_NOISE_MINUS_X0,
    "diffusers": FLOW_VELOCITY_NOISE_MINUS_X0,
    FLOW_VELOCITY_X0_MINUS_NOISE: FLOW_VELOCITY_X0_MINUS_NOISE,
    "triposg": FLOW_VELOCITY_X0_MINUS_NOISE,
}


def normalize_flow_velocity(value):
    if value is None:
        return FLOW_VELOCITY_NOISE_MINUS_X0
    key = str(value).lower().replace("-", "_")
    if key not in _FLOW_VELOCITY_ALIASES:
        valid = ", ".join(sorted(_FLOW_VELOCITY_ALIASES))
        raise ValueError(f"unknown flow_velocity {value}; expected one of: {valid}")
    return _FLOW_VELOCITY_ALIASES[key]


def flow_velocity_to_scheduler_scale(flow_velocity):
    if flow_velocity == FLOW_VELOCITY_NOISE_MINUS_X0:
        return 1.0
    if flow_velocity == FLOW_VELOCITY_X0_MINUS_NOISE:
        return -1.0
    raise ValueError(f"unknown flow_velocity {flow_velocity}")


def get_diffusion_loss_weights(weighting_scheme, sigmas):
    if weighting_scheme == "sigma_sqrt":
        weights = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weights = 2 / (math.pi * bot)
    else:
        weights = torch.ones_like(sigmas)
    return weights


def get_timestep_sampling_density(
    weighting_scheme, N, logit_mean=0.0, logit_std=1.0, mode_scale=1.29
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(N,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(N,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(N,), device="cpu")
    return u


def init_scheduler(name, args):
    scheduler_cls = SCHEDULER_MAPPING[name]
    other_init_keys = []
    for k, sched_cls in SCHEDULER_MAPPING.items():
        init_sig = inspect.signature(sched_cls.__init__)
        keys = [i for i in list(init_sig.parameters) if i != "self"]
        if k == name:
            init_keys = keys
        else:
            other_init_keys += keys
    other_init_keys = set(other_init_keys)

    real_args = {}
    for key, value in args.items():
        if key in init_keys:
            real_args[key] = value
        else:
            # check if key is inherited from other class init, else it is a typo
            assert key in other_init_keys, f"unknown scheduler arg {key}"
    return scheduler_cls(**real_args)


class DiffusionLoss:
    def __init__(self, args_scheduler):
        args_scheduler = copy.deepcopy(args_scheduler)
        scheduler_name = args_scheduler.pop("class_name")
        self.num_inference_steps = args_scheduler.pop("num_inference_timesteps")
        self.flow_velocity = normalize_flow_velocity(
            args_scheduler.pop("flow_velocity", None)
        )
        self.flow_velocity_to_scheduler_scale = flow_velocity_to_scheduler_scale(
            self.flow_velocity
        )
        self.weighting_scheme = args_scheduler.pop("weighting_scheme", None)
        self.low_sigma_sample_prob = float(
            args_scheduler.pop("low_sigma_sample_prob", 0.0)
        )
        self.low_sigma_sample_sigma_max = float(
            args_scheduler.pop("low_sigma_sample_sigma_max", 0.04)
        )
        self.low_sigma_loss_weight = float(
            args_scheduler.pop("low_sigma_loss_weight", 0.0)
        )
        self.low_sigma_loss_sigma_max = float(
            args_scheduler.pop("low_sigma_loss_sigma_max", 0.04)
        )
        self.scheduler = init_scheduler(scheduler_name, args_scheduler)
        if self.flow_velocity != FLOW_VELOCITY_NOISE_MINUS_X0 and not isinstance(
            self.scheduler, FlowMatchEulerDiscreteScheduler
        ):
            raise ValueError("flow_velocity is only supported for FlowMatch schedulers")

    def __call__(self, model, x, condition=None, mask=None, sigma_history=None):
        # scale_factor is applied at the encode chokepoint (encode_in_chunks * sf), so the
        # target x AND the latent context are already in the DM's scaled space here. Do NOT
        # scale x again -- that double-scales the target (encode * sf, then loss * sf) and
        # the model learns sf*-too-large latents (sample()'s 1/sf then leaves them scaled,
        # decoding to an over-sized/distorted mesh). sf=1.0 -> historically a no-op.
        noise = torch.randn_like(x)
        timesteps = self.get_timesteps(len(x), device=x.device)
        x_t = self.add_noise(x, noise, timesteps)
        # Only forward `sigma_history` when set so legacy model classes that
        # don't accept the kwarg keep working unchanged.
        extra = {"sigma_history": sigma_history} if sigma_history is not None else {}
        pred = model(x_t, timesteps=timesteps, condition=condition, mask=mask, **extra)

        # Handle different model output formats
        # TripoSG models return Transformer1DModelOutput with .sample attribute
        # ShapeDiffusionTransformer returns raw tensor
        if hasattr(pred, "sample"):
            pred = pred.sample

        target = self.get_target(x, noise, timesteps)
        return self.compute_weighted_mse(pred, target, timesteps, n_dim=x.ndim)

    def get_timesteps(self, N, device="cuda"):
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            u = get_timestep_sampling_density(
                weighting_scheme=self.weighting_scheme, N=N
            )
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = self.apply_low_sigma_resampling(indices)
            timesteps = self.scheduler.timesteps[indices].to(device=device)

        else:
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (N,),
                device=device,
                dtype=torch.long,
            )
        return timesteps

    def apply_low_sigma_resampling(self, indices, generator=None):
        if self.low_sigma_sample_prob <= 0:
            return indices
        sigmas = self.scheduler.sigmas[:-1].detach().float().cpu()
        low_indices = torch.nonzero(
            (sigmas > 0) & (sigmas <= self.low_sigma_sample_sigma_max),
            as_tuple=False,
        ).flatten()
        if low_indices.numel() == 0:
            return indices
        mask = (
            torch.rand(indices.shape, device="cpu", generator=generator)
            < self.low_sigma_sample_prob
        )
        n_replace = int(mask.sum().item())
        if n_replace == 0:
            return indices
        sampled = low_indices[
            torch.randint(
                0, low_indices.numel(), (n_replace,), device="cpu", generator=generator
            )
        ]
        indices = indices.clone()
        indices[mask] = sampled.to(indices.dtype)
        return indices

    def compute_weighted_mse(self, pred, target, timesteps, n_dim):
        raw_sq = (pred.float() - target.float()).square()
        if self.weighting_scheme is not None:
            sigmas = self.get_sigmas(
                timesteps, n_dim=n_dim, device=pred.device, dtype=pred.dtype
            )
            weights = get_diffusion_loss_weights(self.weighting_scheme, sigmas)
            loss = (raw_sq * weights.expand_as(raw_sq).float()).mean()
        else:
            loss = raw_sq.mean()

        if self.low_sigma_loss_weight > 0:
            sigmas = self.get_sigmas(
                timesteps, n_dim=n_dim, device=pred.device, dtype=pred.dtype
            )
            low_mask = (
                (sigmas > 0) & (sigmas <= self.low_sigma_loss_sigma_max)
            ).float()
            low_mask = low_mask.expand_as(raw_sq)
            denom = low_mask.sum().clamp_min(1.0)
            low_loss = (raw_sq * low_mask).sum() / denom
            loss = loss + self.low_sigma_loss_weight * low_loss
        return loss

    def add_noise(self, x, noise, timesteps):
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = self.get_sigmas(
                timesteps, n_dim=x.ndim, device=x.device, dtype=x.dtype
            )
            x_t = (1.0 - sigmas) * x + sigmas * noise
        else:
            x_t = self.scheduler.add_noise(x, noise, timesteps)
        return x_t

    def get_target(self, x, noise, timesteps):
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            target = self.flow_velocity_to_scheduler_scale * (noise - x)
        elif self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "sample":
            target = x
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x, noise, timesteps)
        else:
            raise ValueError("invalid prediction type for scheduler")
        return target

    def model_output_for_scheduler(self, pred):
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            return self.flow_velocity_to_scheduler_scale * pred
        return pred

    def get_sigmas(self, timesteps, n_dim=4, device="cuda", dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_loss_at_t_with_noise(
        self,
        model,
        x,
        noise,
        condition=None,
        mask=None,
        t_fraction=0.5,
    ):
        """Like compute_loss_at_t, but takes a pre-made noise tensor instead
        of generating one. Use this when noise needs to be sample-deterministic
        (so per-sample loss is invariant to batching / world_size)."""
        # x already in scaled space (encode_in_chunks * sf); do not re-scale (would double).
        n_train = self.scheduler.config.num_train_timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            if not hasattr(self.scheduler, "_fm_eval_inited"):
                self.scheduler.set_timesteps(n_train, device=x.device)
                self.scheduler._fm_eval_inited = True
            ts = self.scheduler.timesteps.to(x.device)
            idx = min(int(t_fraction * len(ts)), len(ts) - 1)
            t_scalar = ts[idx]
        else:
            idx = min(int(t_fraction * n_train), n_train - 1)
            t_scalar = torch.tensor(idx, device=x.device, dtype=torch.long)
        timesteps = t_scalar.expand(len(x)).to(x.device)
        x_t = self.add_noise(x, noise, timesteps)
        pred = model(x_t, timesteps=timesteps, condition=condition, mask=mask)
        if hasattr(pred, "sample"):
            pred = pred.sample
        target = self.get_target(x, noise, timesteps)
        return F.mse_loss(pred, target)

    def compute_loss_at_t(
        self,
        model,
        x,
        condition=None,
        mask=None,
        t_fraction=0.5,
        noise_seed=42,
    ):
        """Deterministic counterpart of __call__: fixed timestep + fixed noise.

        Useful for validation on a fixed timestep and fixed noise sample. See
        compute_loss_at_t_with_noise for the world-size-invariant variant.
        """
        gen = torch.Generator(device=x.device).manual_seed(int(noise_seed))
        noise = torch.empty_like(x).normal_(generator=gen)
        return self.compute_loss_at_t_with_noise(
            model,
            x,
            noise,
            condition=condition,
            mask=mask,
            t_fraction=t_fraction,
        )

    @torch.no_grad()
    def predict_x0_at_t(
        self,
        model,
        x,
        noise,
        condition=None,
        mask=None,
        t_fraction=0.1,
    ):
        """Model's estimate of the clean latent x0, returned in UNSCALED (VAE) space.

        Useful for inspecting the end-of-sequence signal: the inference check runs
        is_near_zero_output(mean|z| < threshold) on the sampled latent, which the
        model emits unscaled. This helper noises a clean target to a fixed timestep,
        runs one forward pass, recovers x0_hat in scaled space, and divides by the
        scale factor so the magnitude is directly comparable to inference.
        """
        # x already scaled at the encode chokepoint; do not re-scale (would double). The
        # scale_factor is still used below to return x0 in native space (matching the
        # inference sampler output) for the EoT threshold comparison.
        scale_factor = getattr(model, "module", model).scale_factor
        n_train = self.scheduler.config.num_train_timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            if not hasattr(self.scheduler, "_fm_eval_inited"):
                self.scheduler.set_timesteps(n_train, device=x.device)
                self.scheduler._fm_eval_inited = True
            ts = self.scheduler.timesteps.to(x.device)
            idx = min(int(t_fraction * len(ts)), len(ts) - 1)
            t_scalar = ts[idx]
        else:
            idx = min(int(t_fraction * n_train), n_train - 1)
            t_scalar = torch.tensor(idx, device=x.device, dtype=torch.long)
        timesteps = t_scalar.expand(len(x)).to(x.device)
        x_t = self.add_noise(x, noise, timesteps)
        pred = model(x_t, timesteps=timesteps, condition=condition, mask=mask)
        if hasattr(pred, "sample"):
            pred = pred.sample
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = self.get_sigmas(
                timesteps, n_dim=x.ndim, device=x.device, dtype=x.dtype
            )
            x0_scaled = x_t - sigmas * self.model_output_for_scheduler(pred)
        elif self.scheduler.config.prediction_type == "v_prediction":
            # x0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v
            ac = self.scheduler.alphas_cumprod.to(x.device)[timesteps]
            while ac.ndim < x.ndim:
                ac = ac.unsqueeze(-1)
            x0_scaled = ac.sqrt() * x_t - (1.0 - ac).sqrt() * pred
        elif self.scheduler.config.prediction_type == "epsilon":
            ac = self.scheduler.alphas_cumprod.to(x.device)[timesteps]
            while ac.ndim < x.ndim:
                ac = ac.unsqueeze(-1)
            x0_scaled = (x_t - (1.0 - ac).sqrt() * pred) / ac.sqrt()
        else:  # "sample"
            x0_scaled = pred
        return x0_scaled / scale_factor


class DiffusionSampler:
    def __init__(self, args_scheduler):
        args_scheduler = copy.deepcopy(args_scheduler)
        scheduler_name = args_scheduler.pop("class_name")
        self.num_inference_steps = args_scheduler.pop("num_inference_timesteps")
        self.flow_velocity = normalize_flow_velocity(
            args_scheduler.pop("flow_velocity", None)
        )
        self.flow_velocity_to_scheduler_scale = flow_velocity_to_scheduler_scale(
            self.flow_velocity
        )
        _ = args_scheduler.pop("weighting_scheme", None)  # unused here
        _ = args_scheduler.pop("low_sigma_sample_prob", None)  # training-only
        _ = args_scheduler.pop("low_sigma_sample_sigma_max", None)  # training-only
        _ = args_scheduler.pop("low_sigma_loss_weight", None)  # training-only
        _ = args_scheduler.pop("low_sigma_loss_sigma_max", None)  # training-only
        self.scheduler = init_scheduler(scheduler_name, args_scheduler)
        if self.flow_velocity != FLOW_VELOCITY_NOISE_MINUS_X0 and not isinstance(
            self.scheduler, FlowMatchEulerDiscreteScheduler
        ):
            raise ValueError("flow_velocity is only supported for FlowMatch schedulers")
        if scheduler_name in [
            "EulerAncestralDiscreteScheduler",
            "EulerDiscreteScheduler",
        ]:
            self.scale_input = True
        else:
            self.scale_input = False

    def model_output_for_scheduler(self, pred):
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            return self.flow_velocity_to_scheduler_scale * pred
        return pred

    def __call__(
        self,
        model,
        latents,
        condition,  # conditions, which is a list of conditions (e.g. [[img_cond1, latent_1], [img_cond2, latent_2]])
        uncondition=None,  # unconditional, which is a list of conditions (e.g. [img_uncond1, latent_uncond])
        cfg_scale=1.0,
        batch_seeds=None,
        cond_weights=None,
        mask=None,
        sigma_history=None,
        sigma_floor=0.0,
        show_progress=False,
        progress_desc="diffusion",
        **kwargs,
    ):
        device = latents.device
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            assert uncondition is not None, "need cond & uncond"

        if batch_seeds is None:
            generator = None
        else:
            generator = [
                torch.Generator(device).manual_seed(int(seed)) for seed in batch_seeds
            ]

        if not isinstance(condition[0], list):
            condition = [condition]

        if cond_weights is None:
            cond_weights = [1.0 / len(condition)] * len(condition)
        else:
            assert len(cond_weights) == len(condition), (
                "cond_weights must match number of conditions"
            )

        # Combine unconditional and all conditional embeddings
        if not isinstance(condition[0], list):
            labels = torch.cat([uncondition, condition[0]]) if do_cfg else condition
        else:
            # If uncondition = [img_uncond, latent_uncond]
            # And condition = [[img_cond1, latent_cond1], [img_cond2, latent_cond2]]
            # Result labels will be:
            # labels = [
            #     torch.cat([img_uncond, img_cond1, img_cond2], dim=0),      # i=0
            #     torch.cat([latent_uncond, latent_cond1, latent_cond2], dim=0)  # i=1
            # ]
            if do_cfg:
                labels = [
                    torch.cat(
                        [uncond] + [cls_label[i] for cls_label in condition],
                        dim=0,
                    )
                    for i, uncond in enumerate(uncondition)
                ]
            else:
                assert len(condition) == 1, (
                    "multi-condition sampling without CFG is ambiguous"
                )
                labels = condition[0]

        self.scheduler.set_timesteps(self.num_inference_steps)
        x_t = latents

        timesteps = self.scheduler.timesteps
        if show_progress and tqdm is not None:
            progress = tqdm(
                timesteps,
                total=len(timesteps),
                desc=progress_desc,
                leave=False,
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            progress = timesteps

        try:
            for t in progress:
                if sigma_floor and float(sigma_floor) > 0:
                    schedule_timesteps = self.scheduler.timesteps.to(latents.device)
                    step_index = (
                        (schedule_timesteps == t.to(latents.device)).nonzero().item()
                    )
                    sigma = self.scheduler.sigmas.to(
                        device=latents.device, dtype=latents.dtype
                    )[step_index]
                    if float(sigma.item()) <= float(sigma_floor):
                        break
                # `labels` is a bare TENSOR for an image-only condition (uncond+cond
                # concatenated on the batch dim, e.g. predict-whole / generate-whole)
                # and a LIST [image_stack, latent_stack] for structured image+part
                # conditions. A tensor must be cloned whole: iterating it would split
                # the CFG batch into a per-row list that forward() misreads as
                # [image_emb, part_latents] (-> "expected 4, got 2" in
                # _prepare_part_conditioning). Lists are cloned element-wise.
                if torch.is_tensor(labels):
                    labels_copy = labels.clone()
                else:
                    labels_copy = [
                        label.clone() if torch.is_tensor(label) else label
                        for label in labels
                    ]
                if do_cfg:
                    x_input = torch.cat(
                        [x_t] * (len(condition) + 1)
                    )  # [B * (1 + N), ...]
                else:
                    x_input = x_t

                t_tensor = t.expand(x_input.shape[0]).to(latents.device)
                # Broadcast sigma_history to match the (uncond + N cond) stacking the
                # sampler does for CFG; same trick as model conditions above.
                if sigma_history is not None:
                    sh_input = (
                        torch.cat([sigma_history] * (len(condition) + 1))
                        if do_cfg
                        else sigma_history
                    )
                    pred = model(
                        x_input,
                        condition=labels_copy,
                        timesteps=t_tensor,
                        mask=mask,
                        sigma_history=sh_input,
                    )
                else:
                    pred = model(
                        x_input, condition=labels_copy, timesteps=t_tensor, mask=mask
                    )

                # Handle different model output formats
                # TripoSG models return Transformer1DModelOutput with .sample attribute
                # ShapeDiffusionTransformer returns raw tensor
                if hasattr(pred, "sample"):
                    pred = pred.sample

                if do_cfg:
                    B = latents.shape[0]
                    pred_uncond = pred[:B]
                    pred_conds = pred[B:].chunk(len(condition))  # list of tensors
                    # weighted_sum = sum(
                    #     w * (pc - pred_uncond) for w, pc in zip(cond_weights, pred_conds)
                    # )
                    if len(condition) == 1:
                        weighted_sum = cond_weights[0] * (pred_conds[0] - pred_uncond)
                    else:
                        weighted_sum = cond_weights[0] * (
                            pred_conds[0] - pred_conds[1]
                        ) + cond_weights[1] * (pred_conds[1] - pred_uncond)
                    pred = pred_uncond + cfg_scale * weighted_sum

                out = self.scheduler.step(
                    model_output=self.model_output_for_scheduler(pred),
                    timestep=t,
                    sample=x_t,
                    generator=generator,
                )
                x_t = out.prev_sample
        finally:
            if progress is not timesteps and hasattr(progress, "close"):
                progress.close()

        return x_t
