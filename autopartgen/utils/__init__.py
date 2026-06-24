from functools import wraps
from random import (
    getstate as rand_get_state,
    seed as rand_seed,
    setstate as rand_set_state,
)

from numpy.random import (
    get_state as np_get_state,
    seed as np_seed,
    set_state as np_set_state,
)
from torch import (
    cuda as torch_cuda,
    get_rng_state as torch_get_state,
    manual_seed as torch_seed,
    set_rng_state as torch_set_state,
)


class use_seed:
    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int) and seed >= 0
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.rand_state = rand_get_state()
            self.np_state = np_get_state()
            self.torch_state = torch_get_state()
            self.cuda_state = (
                torch_cuda.get_rng_state_all() if torch_cuda.is_available() else None
            )
            rand_seed(self.seed)
            np_seed(self.seed)
            torch_seed(self.seed)
            if torch_cuda.is_available():
                torch_cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            rand_set_state(self.rand_state)
            np_set_state(self.np_state)
            torch_set_state(self.torch_state)
            if self.cuda_state is not None:
                torch_cuda.set_rng_state_all(self.cuda_state)

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop("seed", None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper
