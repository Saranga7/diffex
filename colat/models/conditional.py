import torch

from colat.models.abstract import Model
from colat.utils.net_utils import create_mlp


class LinearConditional(Model):
    """K directions linearly conditional on latent code"""

    def __init__(
        self,
        k: int,
        size: int,
        alpha: float = 0.1,
        normalize: bool = True,
        bias: bool = False,
        batchnorm: bool = False,
    ) -> None:
        super().__init__(k=k, size=size, alpha=alpha, normalize=normalize)
        self.k = k
        self.size = size

        # make mlp net
        self.nets = torch.nn.ModuleList()

        for i in range(k):
            net = create_mlp(
                depth=1,
                in_features=size,
                middle_features=-1,
                out_features=size,
                bias=bias,
                batchnorm=batchnorm,
                final_norm=batchnorm,
            )
            self.nets.append(net)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size])
        z = z.repeat(
            (
                self.k,
                1,
                1,
            )
        )

        # calculate directions
        dz = []
        for i in range(self.k):
            res_dz = self.nets[i](z[i, ...])
            res_dz = self.post_process(res_dz)
            dz.append(res_dz)

        dz = torch.stack(dz)

        #  add directions
        z = z + dz

        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.nets[k](z))


class NonlinearConditional(Model):
    """K directions nonlinearly conditional on latent code"""

    def __init__(
        self,
        k: int,
        size: int,
        depth: int,
        alpha: float = 0.1,
        normalize: bool = True,
        bias: bool = True,
        batchnorm: bool = True,
        final_norm: bool = False,
    ) -> None:
        super().__init__(k=k, size=size, alpha=alpha, normalize=normalize)
        self.k = k
        self.size = size

        # make mlp net
        self.nets = torch.nn.ModuleList()

        for i in range(k):
            net = create_mlp(
                depth=depth,
                in_features=size,
                middle_features=size,
                out_features=size,
                bias=bias,
                batchnorm=batchnorm,
                final_norm=final_norm,
            )
            self.nets.append(net)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size])
        z = z.repeat(
            (
                self.k,
                1,
                1,
            )
        )

        #  calculate directions
        dz = []
        for i in range(self.k):
            res_dz = self.nets[i](z[i, ...])
            res_dz = self.post_process(res_dz)
            dz.append(res_dz)

        dz = torch.stack(dz)
        
       
        #  Added by Anis
        non_diag_sum = 0.0

        for batch_idx in range(dz.shape[1]):  # Iterate over each batch element
            dz_batch = dz[:, batch_idx, :]  # Shape: [k, size]
            dz_batch = dz_batch - dz_batch.mean(dim=1, keepdim=True)  # Center the directions

            # Compute the covariance matrix
            cov_matrix = torch.matmul(dz_batch, dz_batch.t()) / (self.size - 1)  # Shape: [k, k]

            # Extract non-diagonal elements (upper or lower triangular excluding the diagonal)
            non_diag_elements = cov_matrix[~torch.eye(self.k, dtype=bool)]  # Extract non-diagonal elements

            # Sum the non-diagonal elements
            non_diag_sum += non_diag_elements.pow(2).sum()

        #  add directions
        z = z + dz

        return torch.reshape(z, [-1, self.size]), non_diag_sum

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.nets[k](z))
