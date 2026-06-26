# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Part-conditioned TripoSG DiT used by AutoPartGen."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from autopartgen.diffusion.diffusion import DiffusionSampler
from autopartgen.layers.embedding import PositionalEmbedding
from autopartgen.models.triposg.attention_processor import (
    FusedTripoSGAttnProcessor2_0,
    TripoSGAttnProcessor2_0,
)
from autopartgen.models.triposg.transformers.modeling_outputs import (
    Transformer1DModelOutput,
)
from autopartgen.models.triposg.transformers.triposg_transformer import FP32LayerNorm
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import LayerNorm
from diffusers.utils import (
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    USE_PEFT_BACKEND,
)
from torch import nn

logger = logging.get_logger(__name__)


class PartgenDiTBlock(nn.Module):
    """Transformer block with image and optional part cross-attention."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dims: Union[int, List[int]] = 1536,
        use_self_attention: bool = True,
        use_part_cross_attention: bool = True,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,
        skip_norm_last: bool = False,
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()

        if isinstance(cross_attention_dims, int):
            cross_attention_dims = [cross_attention_dims]

        self.use_self_attention = use_self_attention
        self.use_part_cross_attention = use_part_cross_attention
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last
        self.num_cross_attentions = len(cross_attention_dims)

        # Self-attention over latent tokens.
        if use_self_attention:
            self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # Image cross-attention. This branch is always present.
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dims[0],
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-6,
            bias=qkv_bias,
            processor=TripoSGAttnProcessor2_0(),
        )

        # Previous-part cross-attention, used only in early blocks.
        if use_part_cross_attention and len(cross_attention_dims) > 1:
            self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            self.attn3 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dims[1],
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )
        else:
            self.norm3 = None
            self.attn3 = None

        # Feed-forward path.
        self.norm_ff = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # Decoder-side long skip.
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        attention_kwargs = attention_kwargs or {}

        if isinstance(encoder_hidden_states, (list, tuple)):
            image_context = encoder_hidden_states[0]
            part_context = (
                encoder_hidden_states[1] if len(encoder_hidden_states) > 1 else None
            )
        else:
            image_context = encoder_hidden_states
            part_context = None

        # Decoder skip path.
        if self.skip_linear is not None:
            cat = torch.cat(
                (
                    [skip, hidden_states]
                    if self.skip_concat_front
                    else [hidden_states, skip]
                ),
                dim=-1,
            )
            if self.skip_norm_last:
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=image_context,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        if self.attn3 is not None and part_context is not None:
            hidden_states = hidden_states + self.attn3(
                self.norm3(hidden_states),
                encoder_hidden_states=part_context,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

        mlp_inputs = self.norm_ff(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states


class StackedRandomGenerator:
    """Random generator for deterministic sampling across batch."""

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


class PartgenTripoSGDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """TripoSG DiT with autoregressive part conditioning."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1536,
        latent_dim: int = 64,
        num_latents: int = 2048,
        token_cross_attention_layer: int = 12,
        use_part_position_embedding: bool = True,
        latent_on_spatial: bool = False,
        mlp_encode_cond: bool = True,
        scale_factor: float = 1.0,
        part_as_concat: bool = False,
        args_scheduler: Optional[Dict] = None,
        use_diffusion_forcing: bool = False,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0
        self.num_latents = num_latents
        self.input_dim = in_channels
        self.scale_factor = scale_factor
        self.token_cross_attention_layer = token_cross_attention_layer
        self.latent_on_spatial = latent_on_spatial
        self.mlp_encode_cond = mlp_encode_cond
        self.latent_dim = latent_dim
        self.cross_attention_dim = cross_attention_dim

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )

        self.use_diffusion_forcing = use_diffusion_forcing
        if use_diffusion_forcing:
            self.sigma_history_proj = TimestepEmbedding(
                timestep_input_dim,
                time_embed_dim,
                act_fn="gelu",
                out_dim=self.inner_dim,
            )
            self.sigma_per_part_proj = TimestepEmbedding(
                timestep_input_dim,
                time_embed_dim,
                act_fn="gelu",
                out_dim=latent_dim,
            )
        else:
            # Keep the checkpoint schema unchanged when diffusion forcing is off.
            self.sigma_history_proj = None
            self.sigma_per_part_proj = None
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=True)

        # Part index embeddings are added before part cross-attention.
        if use_part_position_embedding:
            self.part_position_embedding = PositionalEmbedding(self.inner_dim)
            self.map_position = nn.Linear(self.inner_dim, latent_dim)
            self.sot_token = torch.zeros(size=(num_latents, latent_dim))
            self.eot_token = torch.zeros(size=(num_latents, latent_dim))
        else:
            self.part_position_embedding = None
            self.map_position = None
            self.sot_token = None
            self.eot_token = None

        # Optional projection for packed image/mask conditioning.
        if mlp_encode_cond:
            if part_as_concat:
                context_in_dim = cross_attention_dim
            else:
                context_in_dim = cross_attention_dim * 2
            self.cond_proj = nn.Sequential(
                nn.Linear(context_in_dim, cross_attention_dim),
                nn.GELU(),
                nn.LayerNorm(cross_attention_dim),
                nn.Linear(cross_attention_dim, cross_attention_dim),
            )
        else:
            self.cond_proj = None

        self.blocks = nn.ModuleList()
        for layer in range(num_layers):
            # Early blocks attend to previous parts; later blocks use image only.
            use_part_xattn = (
                layer < token_cross_attention_layer and not latent_on_spatial
            )

            if use_part_xattn:
                block_cross_dims = [cross_attention_dim, latent_dim]
            else:
                block_cross_dims = [cross_attention_dim]

            self.blocks.append(
                PartgenDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    cross_attention_dims=block_cross_dims,
                    use_self_attention=True,
                    use_part_cross_attention=use_part_xattn,
                    activation_fn="gelu",
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,
                    qk_norm=True,
                    qkv_bias=False,
                )
            )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

        if args_scheduler is not None:
            self.sampler = DiffusionSampler(args_scheduler)
        else:
            self.sampler = None

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load native weights or remap original TripoSG cross-attention keys."""
        has_triposg_keys = any("attn_cross.0." in key for key in state_dict)
        verbose = os.environ.get("APG_VERBOSE_LOAD", "0") == "1"

        if verbose:
            logger.info(
                "Loading %d keys into PartgenTripoSGDiTModel "
                "(triposg_key_layout=%s)",
                len(state_dict),
                has_triposg_keys,
            )

        if has_triposg_keys:
            remapped_state_dict = {}
            remapped_count = 0
            norm3_to_ff_count = 0
            for key, value in state_dict.items():
                new_key = key
                if "attn_cross.0." in key:
                    new_key = key.replace("attn_cross.0.", "attn2.")
                    remapped_count += 1
                if "norm_cross.0" in new_key:
                    new_key = new_key.replace("norm_cross.0", "norm2")
                if (
                    ".norm3." in new_key
                    or new_key.endswith(".norm3.weight")
                    or new_key.endswith(".norm3.bias")
                ):
                    new_key = (
                        new_key.replace(".norm3.", ".norm_ff.")
                        .replace(".norm3.weight", ".norm_ff.weight")
                        .replace(".norm3.bias", ".norm_ff.bias")
                    )
                    norm3_to_ff_count += 1
                remapped_state_dict[new_key] = value

            if verbose:
                logger.info(
                    "Remapped TripoSG keys: attn_cross.0 -> attn2 (%d), "
                    "norm3 -> norm_ff (%d)",
                    remapped_count,
                    norm3_to_ff_count,
                )
                sample_keys = [k for k in remapped_state_dict if "attn2" in k][:5]
                logger.debug("Sample remapped keys: %s", sample_keys)

            state_dict = remapped_state_dict

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        if (
            verbose
            and hasattr(result, "missing_keys")
            and hasattr(result, "unexpected_keys")
        ):
            attn3_missing = [k for k in result.missing_keys if "attn3" in k]
            logger.info(
                "State dict load result: missing=%d unexpected=%d attn3_missing=%d",
                len(result.missing_keys),
                len(result.unexpected_keys),
                len(attn3_missing),
            )

        return result

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4
            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please use 'positional'."
            )
        return time_embed_dim, timestep_input_dim

    def _prepare_part_conditioning(
        self,
        previous_parts: torch.Tensor,
        device: torch.device,
        sigma_per_part: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Prepare previous-part tokens for the part cross-attention stream."""
        if previous_parts is None:
            return None

        B, num_parts, num_latents, latent_dim = previous_parts.shape

        if self.part_position_embedding is not None:
            positions = torch.arange(num_parts, device=device)
            part_pe = self.part_position_embedding(positions)
            part_pe = torch.nn.functional.silu(self.map_position(part_pe))
            part_pe = part_pe.unsqueeze(0).unsqueeze(2)
            previous_parts = previous_parts + part_pe

        if self.sigma_per_part_proj is not None and sigma_per_part is not None:
            sh_flat = (sigma_per_part.to(device=device).float() * 1000.0).reshape(-1)
            sh_emb = self.time_embed(sh_flat).to(previous_parts.dtype)
            sh_emb = self.sigma_per_part_proj(sh_emb)
            sh_emb = sh_emb.view(B, num_parts, 1, latent_dim)
            previous_parts = previous_parts + sh_emb

        part_tokens = previous_parts.reshape(B, num_parts * num_latents, latent_dim)

        return part_tokens

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """Returns dict of attention processors indexed by weight name."""
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        """Sets the attention processor to use."""
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed with {len(processor)} entries, "
                f"but the model has {count} attention layers."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """Sets the default attention implementation."""
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def fuse_qkv_projections(self):
        """Enables fused QKV projections."""
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models with "
                    "added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        """Disables fused QKV projection if enabled."""
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: Union[int, float, torch.LongTensor],
        condition: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        mask: Optional[torch.Tensor] = None,  # noqa: F841 - Reserved for future attention masking
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        sigma_history: Optional[torch.Tensor] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` is ineffective "
                    "without the PEFT backend."
                )

        _, N, _ = hidden_states.shape
        original_shape = hidden_states.shape
        device = hidden_states.device

        # Split image context and autoregressive part history.
        if isinstance(condition, list):
            image_emb = condition[0]
            part_latents = condition[1]

            sigma_per_part = None
            if sigma_history is not None and sigma_history.dim() == 2:
                sigma_per_part = sigma_history

            part_tokens = self._prepare_part_conditioning(
                part_latents, device=device, sigma_per_part=sigma_per_part
            )
        else:
            image_emb = condition
            part_tokens = None

        if self.cond_proj is not None and image_emb is not None:
            image_emb = self.cond_proj(image_emb)

        # Legacy spatial-history path; release inference uses attn3.
        if self.latent_on_spatial and part_tokens is not None:
            part_tokens_spatial = nn.functional.linear(
                part_tokens,
                self.proj_in.weight[:, : self.latent_dim],
            )
            hidden_states = torch.cat([hidden_states, part_tokens_spatial], dim=1)

        temb = self.time_embed(timesteps).to(hidden_states.dtype)
        temb = self.time_proj(temb)

        # Add a global summary of history noise to the timestep embedding.
        if self.sigma_history_proj is not None:
            if sigma_history is None:
                sh_scalar = torch.zeros(
                    hidden_states.shape[0], device=device, dtype=hidden_states.dtype
                )
            else:
                sh = sigma_history.to(device=device)
                sh_scalar = sh.float().mean(dim=1) if sh.dim() == 2 else sh.float()
            sh_t = sh_scalar * 1000.0
            sh_emb = self.time_embed(sh_t).to(hidden_states.dtype)
            sh_emb = self.sigma_history_proj(sh_emb)
            temb = temb + sh_emb

        temb = temb.unsqueeze(dim=1)

        # Prepend the time token before the transformer stack.
        hidden_states = self.proj_in(hidden_states)

        hidden_states = torch.cat([temb, hidden_states], dim=1)

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()

            # Only early blocks consume the part-history context.
            if layer < self.token_cross_attention_layer and not self.latent_on_spatial:
                encoder_hidden_states = [image_emb, part_tokens]
            else:
                encoder_hidden_states = [image_emb]

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    skip,
                    attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                )

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        # Remove the time token and any extra spatial-history tokens.
        hidden_states = hidden_states[:, -N:]
        if self.latent_on_spatial and part_tokens is not None:
            hidden_states = hidden_states[:, : original_shape[1], ...]
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    @torch.no_grad()
    def sample_with_condition(
        self,
        cond,
        batch_seeds=None,
        uncond=None,
        cfg_scale=1.0,
        cond_weights=None,
        mask=None,
        sigma_history=None,
        sigma_floor=0.0,
        show_progress=False,
        progress_desc="diffusion",
    ):
        """Sample with image and part-history conditioning."""
        device = batch_seeds.device
        batch_size = batch_seeds.shape[0]

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, self.num_latents, self.input_dim], device=device
        )
        pred = self.sampler(
            model=self,
            latents=latents,
            condition=cond,
            uncondition=uncond,
            cfg_scale=cfg_scale,
            batch_seeds=batch_seeds,
            cond_weights=cond_weights,
            mask=mask,
            sigma_history=sigma_history,
            sigma_floor=sigma_floor,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        return 1.0 / self.scale_factor * pred
