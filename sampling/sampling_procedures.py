# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple, TypeVar, Generic, Optional

import torch
from torch import Tensor
from torch.distributions import Distribution

from ..ops import Manifold
from ..distributions import RadiusVonMisesFisher, HypersphericalUniform, WrappedNormal, EuclideanNormal
from ..distributions import EuclideanUniform, RadiusProjectedVonMisesFisher, RiemannianNormal
from ..ops import spherical_projected as SP

Q = TypeVar('Q', bound=Distribution)
P = TypeVar('P', bound=Distribution)


class SamplingProcedure(Generic[Q, P]):

    def __init__(self, component, manifold: Manifold, scalar_parametrization: bool) -> None:
        self._component = component
        self._manifold = manifold
        self._scalar_parametrization = scalar_parametrization

    @property
    def scalar_parametrization(self) -> bool:
        return self._scalar_parametrization

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[Q, P]:
        raise NotImplementedError

    def kl_loss(self, q_z: Q, p_z: P, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return torch.distributions.kl.kl_divergence(q_z, p_z)

    def rsample_log_probs(self, sample_shape: torch.Size, q_z: Q, p_z: P) -> Tuple[Tensor, Tensor, Tensor]:
        z, log_q_z_x_ = q_z.rsample_log_prob(sample_shape)
        log_p_z_ = p_z.log_prob(z)
        return z, log_q_z_x_, log_p_z_

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def prior(self, shape, device):
        raise NotImplementedError


class SphericalVmfProcedure(SamplingProcedure[RadiusVonMisesFisher, HypersphericalUniform]):

    def __init__(self, component, manifold: Manifold, scalar_parametrization: bool) -> None:
        if not scalar_parametrization:
            raise ValueError("Spherical VMF only works with scalar scale.")
        super().__init__(component, manifold, scalar_parametrization)

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[RadiusVonMisesFisher, HypersphericalUniform]:
        if std.size(-1) > 1:
            assert std.mean(dim=-1).allclose(std[..., 0])
            std = std[..., 0:1]
        q_z = RadiusVonMisesFisher(z_mean, std, radius=self._manifold.radius)

        # True dim, not ambient space dim:
        p_z = HypersphericalUniform(z_mean.shape[-1] - 1, device=z_mean.device)
        return q_z, p_z

    def prior(self, shape, device):
        return HypersphericalUniform(shape.shape[-1] - 1, device=device)


class ProjectedSphericalVmfProcedure(SamplingProcedure[RadiusProjectedVonMisesFisher, HypersphericalUniform]):

    def __init__(self, component,  manifold: Manifold, scalar_parametrization: bool) -> None:
        if not scalar_parametrization:
            raise ValueError("Projected spherical VMF only works with scalar scale.")
        super().__init__(component, manifold, scalar_parametrization)

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[RadiusProjectedVonMisesFisher, HypersphericalUniform]:
        if std.size(-1) > 1:
            assert std.mean(dim=-1).allclose(std[..., 0])
            std = std[..., 0:1]
        radius = self._manifold.radius
        z_mean_h = SP.projected_to_spherical(z_mean, radius)
        q_z = RadiusProjectedVonMisesFisher(z_mean_h, std, radius=radius)

        # True dim, not ambient space dim:
        p_z = HypersphericalUniform(z_mean_h.shape[-1] - 1, device=z_mean.device)
        return q_z, p_z

    def prior(self, shape, device):
        return HypersphericalUniform(shape.shape[-1] - 1, device=device)


class WrappedNormalProcedure(SamplingProcedure[WrappedNormal, WrappedNormal]):

    def __init__(self, component, manifold: Manifold, scalar_parametrization: bool) -> None:
        super().__init__(component, manifold, scalar_parametrization)
        self._component._mu_0 = torch.nn.Parameter(
            self._manifold.exp_map_mu0(torch.rand((1, self._component.true_dim))),
            requires_grad=True
        )
        if scalar_parametrization:
            self._component._std_0 = torch.nn.Parameter(torch.rand((1, 1)).sqrt(), requires_grad=True)
        else:
            self._component._std_0 = torch.nn.Parameter(torch.rand((1, self._component.true_dim)).sqrt(), requires_grad=True)

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[WrappedNormal, WrappedNormal]:
        q_z = WrappedNormal(z_mean, std, manifold=self._manifold)
        p_z = WrappedNormal(
            self._component._mu_0.expand_as(z_mean),
            self._component._std_0.expand_as(std),
            manifold=self._manifold
        )
        return q_z, p_z

    def prior(self, shape, device):
        return WrappedNormal(
            self._component._mu_0.expand(shape),
            self._component._std_0.expand(*shape[:-1], -1),
            manifold=self._manifold
        )

    def kl_loss(self, q_z: WrappedNormal, p_z: WrappedNormal, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        logqz, logpz = self._log_prob(q_z, p_z, z, data)
        KLD = logqz - logpz
        return KLD

    def rsample_log_probs(self, sample_shape: torch.Size, q_z: WrappedNormal,
                          p_z: WrappedNormal) -> Tuple[Tensor, Tensor, Tensor]:
        z, posterior_parts = q_z.rsample_with_parts(sample_shape)
        log_q_z_x_, log_p_z_ = self._log_prob(q_z, p_z, z, posterior_parts)
        return z, log_q_z_x_, log_p_z_

    def _log_prob(self, q_z: WrappedNormal, p_z: WrappedNormal, z: Tensor,
                  posterior_parts: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        log_q_z_x_ = q_z.log_prob_from_parts(z, posterior_parts)
        log_p_z_ = p_z.log_prob(z)
        return log_q_z_x_, log_p_z_


class EuclideanConstantProcedure(SamplingProcedure[EuclideanUniform, EuclideanUniform]):

    def __init__(self,
                 component,
                 manifold: Manifold,
                 scalar_parametrization: bool,
                 dim: int,
                 const: Optional[torch.Tensor] = None,
                 eps: Optional[torch.Tensor] = None) -> None:
        super().__init__(component, manifold, scalar_parametrization)
        if const is None:
            const = torch.zeros(dim)
        self.const = const
        if eps is None:
            eps = torch.ones(dim) * 1e-4
        self.eps = eps

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[EuclideanUniform, EuclideanUniform]:
        pt = z_mean
        diff = self.eps * std
        return EuclideanUniform(pt - diff, pt + diff), EuclideanUniform(self.const - self.eps, self.const + self.eps)

    def prior(self, shape, device):
        return EuclideanUniform(self.const - self.eps, self.const + self.eps)

    def kl_loss(self, q_z: EuclideanUniform, p_z: EuclideanUniform, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        res = super().kl_loss(q_z, p_z, z, data)
        return res.sum(dim=-1)


class EuclideanNormalProcedure(SamplingProcedure[EuclideanNormal, EuclideanNormal]):

    def __init__(self, component, manifold: Manifold, scalar_parametrization: bool) -> None:
        super().__init__(component, manifold, scalar_parametrization)
        self._component._mu_0 = torch.nn.Parameter(
            torch.rand((1, self._component.true_dim)),
            requires_grad=True
        )
        if scalar_parametrization:
            self._component._std_0 = torch.nn.Parameter(torch.rand((1, 1)).sqrt(), requires_grad=True)
        else:
            self._component._std_0 = torch.nn.Parameter(torch.rand((1, self._component.true_dim)).sqrt(), requires_grad=True)

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[EuclideanNormal, EuclideanNormal]:
        q_z = EuclideanNormal(z_mean, std)
        p_z = EuclideanNormal(self._component._mu_0.expand_as(z_mean), self._component._std_0.expand_as(std))
        return q_z, p_z

    def prior(self, shape, device):
        return EuclideanNormal(
            self._component._mu_0.expand(shape),
            self._component._std_0.expand(*shape[:-1], -1)
        )

    def kl_loss(self, q_z: EuclideanNormal, p_z: EuclideanNormal, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        # res = super().kl_loss(q_z, p_z, z, data)
        log_var_q = torch.nn.Softplus(beta=0.5)(torch.pow(q_z.stddev, 2))
        log_var_p = torch.nn.Softplus(beta=0.5)(torch.pow(p_z.stddev, 2))
        res1 = 0.5 * (log_var_p - log_var_q  # log s_p,i^2 / s_q_i^2
               + torch.exp(log_var_q) / torch.exp(
                    log_var_p)  # s_q_i^2 / s_p_i^2
               + torch.pow(q_z.mean - p_z.mean, 2) / torch.exp(
                    log_var_p)  # (m_p_i - m_q_i)^2 / s_p_i^2
               - 1)
        # return res.sum(dim=-1)
        return res1.sum(dim=-1)


class RiemannianNormalProcedure(SamplingProcedure[RiemannianNormal, RiemannianNormal]):

    def __init__(self, component,  manifold: Manifold, scalar_parametrization: bool) -> None:
        if not scalar_parametrization:
            raise ValueError("Riemannian Normal only works with scalar scale.")
        super().__init__(component, manifold, scalar_parametrization)

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[RiemannianNormal, RiemannianNormal]:
        if std.size(-1) > 1:
            assert std.mean(dim=-1).allclose(std[..., 0])
            std = std[..., 0:1]

        q_z = RiemannianNormal(z_mean, std, manifold=self._manifold)

        mu_0 = self._manifold.mu_0(z_mean.shape, device=z_mean.device)
        std_0 = torch.ones_like(std, device=z_mean.device)
        p_z = RiemannianNormal(mu_0, std_0, manifold=self._manifold)
        return q_z, p_z

    def prior(self, shape, device):
        mu_0 = self._manifold.mu_0(shape, device=device)
        std_0 = torch.ones((*shape[:-1], 1), device=device)
        return RiemannianNormal(mu_0, std_0, manifold=self._manifold)

    def kl_loss(self, q_z: RiemannianNormal, p_z: RiemannianNormal, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        logqz = q_z.log_prob(z)
        logpz = p_z.log_prob(z)
        KLD = logqz - logpz
        return KLD


class UniversalSamplingProcedure(SamplingProcedure[Q, P]):

    def __init__(self, component,  manifold: Manifold, scalar_parametrization: bool) -> None:
        super().__init__(component, manifold, scalar_parametrization)
        self._manifold = manifold
        self._sampling_procedures = {
            -1: WrappedNormalProcedure(self._manifold._manifolds[-1], scalar_parametrization),
            0: EuclideanNormalProcedure(self._manifold._manifolds[0], scalar_parametrization),
            1: WrappedNormalProcedure(self._manifold._manifolds[1], scalar_parametrization),
        }

    @property
    def sampling_procedure(self) -> SamplingProcedure[Q, P]:
        return self._sampling_procedures[self._manifold._choice]

    def reparametrize(self, z_mean: Tensor, std: Tensor) -> Tuple[Q, P]:
        return self.sampling_procedure.reparametrize(z_mean, std)

    def prior(self, shape, device):
        return self._sampling_procedures[self._manifold._choice].prior(shape, device)

    def kl_loss(self, q_z: Q, p_z: P, z: Tensor, data: Tuple[Tensor, ...]) -> Tensor:
        return self.sampling_procedure.kl_loss(q_z, p_z, z, data)

    def rsample_log_probs(self, sample_shape: torch.Size, q_z: Q, p_z: P) -> Tuple[Tensor, Tensor, Tensor]:
        return self.sampling_procedure.rsample_log_probs(sample_shape, q_z, p_z)
