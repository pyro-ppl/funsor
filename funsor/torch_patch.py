import torch
from pyro.distributions.torch_patch import patch_dependency

assert torch.__version__.startswith('1.')


# Support batched inputs to .cholesky_inverse().
try:
    torch.eye(2).expand(2, 2, 2).cholesky_inverse()
except RuntimeError:
    @patch_dependency('torch.Tensor.cholesky_inverse')
    def _cholesky_inverse(self):
        if self.dim() == 2:
            return _cholesky_inverse._pyro_unpatched(self)
        return torch.eye(self.size(-1)).expand(self.size()).cholesky_solve(self)

    # Check that patch works.
    torch.eye(2).expand(2, 2, 2).cholesky_inverse()


__all__ = []
