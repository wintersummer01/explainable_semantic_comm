import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GaussianDensityEstimation', 'GaussianKernalDensityEstimation', 'ShiftedLogKDE']

class GaussianDensityEstimation(nn.Module):
    def __init__(self, num_symbols=512):
        super().__init__()
        self.eps = 1e-30
        self.sigma_eps = 1e-5

    def prob(self, x, mu=0, sigma=1):
        """
        Calculates probability density value of 'x' in Gaussian distribution
        """
        if torch.is_tensor(sigma):
            sigma = torch.clamp(sigma, min=self.sigma_eps)
        else:
            sigma = max(float(sigma), self.sigma_eps)

        exponent = -0.5 * ((x - mu) / sigma) ** 2
        pdf = 1.0 / (sigma * (2 * np.pi) ** 0.5) * torch.exp(exponent)
        return pdf

    def entropy(self, x, mu=0, sigma=1):
        """
        Calculates Gaussian cross entropy of x
        """
        pdf = self.prob(x, mu, sigma)
        return -torch.mean(torch.log(pdf + self.eps), dim=-1)


class GaussianKernalDensityEstimation(nn.Module):
    def __init__(
        self,
        chunk_size=512,
        kernel_size=None,
        grid_size=4096,
        cut=5.0,
        exact_query_max_points=2048,
    ):
        """
        chunk_size:
            pairwise exact evaluation 시 사용할 chunk 크기
        kernel_size:
            bandwidth h. None이면 Scott/Silverman 스타일 h 사용
        grid_size:
            내부 적분 / interpolation 용 grid 개수
        cut:
            grid 범위를 min/max 바깥으로 몇 h 만큼 넓힐지
        exact_query_max_points:
            query point가 이 개수 이하이면 prob(x, x_point)는 exact chunked KDE 계산
            그보다 크면 grid interpolation 사용
        """
        super().__init__()
        self.eps = 1e-30
        self.h = kernel_size
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.cut = cut
        self.exact_query_max_points = exact_query_max_points
        self.normal = GaussianDensityEstimation(chunk_size)

    # ---------------------------
    # helpers
    # ---------------------------
    def _ensure_2d(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True
        elif x.dim() != 2:
            raise ValueError(f'Expected 1D or 2D tensor, got shape={tuple(x.shape)}')
        return x, squeezed

    def _get_query_rows(self, x_point, batch_size, device, dtype):
        if not torch.is_tensor(x_point):
            x_point = torch.as_tensor(x_point, dtype=dtype, device=device)
        else:
            x_point = x_point.to(device=device, dtype=dtype)

        if x_point.dim() == 1:
            return [x_point] * batch_size
        if x_point.dim() == 2:
            if x_point.shape[0] == 1:
                return [x_point[0]] * batch_size
            if x_point.shape[0] == batch_size:
                return [x_point[i] for i in range(batch_size)]

        raise ValueError(
            f'x_point must be shape [M], [1, M], or [B, M]. Got shape={tuple(x_point.shape)}'
        )

    def _get_bandwidth(self, x_row):
        if self.h is not None:
            if torch.is_tensor(self.h):
                h = self.h.to(device=x_row.device, dtype=x_row.dtype)
            else:
                h = torch.tensor(float(self.h), device=x_row.device, dtype=x_row.dtype)
            return torch.clamp(h, min=self.normal.sigma_eps)

        n = max(int(x_row.numel()), 1)
        std = torch.std(x_row, unbiased=False)
        h = 1.06 * std * (n ** (-0.2))
        return torch.clamp(h, min=self.normal.sigma_eps)

    def _chunked_exact_prob_1d(self, x_row, query_row, h):
        """
        Exact KDE at query_row with chunked pairwise accumulation.
        Complexity: O(N * M), but memory-safe.
        """
        x_row = x_row.flatten()
        query_row = query_row.flatten()

        n = x_row.numel()
        out = torch.empty_like(query_row)
        const = 1.0 / (n * h * math.sqrt(2.0 * math.pi))

        for q_start in range(0, query_row.numel(), self.chunk_size):
            q = query_row[q_start:q_start + self.chunk_size].unsqueeze(-1)   # [Mq, 1]
            acc = torch.zeros(q.shape[0], device=x_row.device, dtype=x_row.dtype)

            for x_start in range(0, n, self.chunk_size):
                x_chunk = x_row[x_start:x_start + self.chunk_size].unsqueeze(0)  # [1, Nx]
                z = (q - x_chunk) / h
                acc = acc + torch.exp(-0.5 * z * z).sum(dim=-1)

            out[q_start:q_start + q.shape[0]] = const * acc

        return out

    def _make_grid_1d(self, x_row, h):
        x_min = torch.min(x_row) - self.cut * h
        x_max = torch.max(x_row) + self.cut * h

        if torch.isclose(x_min, x_max):
            x_min = x_min - h
            x_max = x_max + h

        grid = torch.linspace(
            x_min,
            x_max,
            self.grid_size,
            device=x_row.device,
            dtype=x_row.dtype,
        )
        return grid

    def _kde_on_grid_1d(self, x_row, grid, h):
        """
        Histogram + Gaussian convolution 방식의 fast KDE on grid
        """
        x_row = x_row.flatten()
        grid = grid.flatten()

        dx = grid[1] - grid[0]
        g = grid.numel()

        left = grid[0] - dx / 2
        right = grid[-1] + dx / 2
        edges = torch.linspace(left, right, g + 1, device=grid.device, dtype=grid.dtype)

        idx = torch.bucketize(x_row, edges) - 1
        idx = idx.clamp(0, g - 1)

        mass = torch.bincount(idx, minlength=g).to(grid.dtype)
        mass = mass / mass.sum().clamp_min(1.0)

        radius = max(1, int(torch.ceil(self.cut * h / dx).item()))
        offsets = torch.arange(-radius, radius + 1, device=grid.device, dtype=grid.dtype)

        kernel = torch.exp(-0.5 * ((offsets * dx) / h) ** 2) / (h * math.sqrt(2.0 * math.pi))
        kernel = kernel / (kernel.sum() * dx).clamp_min(self.eps)

        pdf = F.conv1d(
            mass.view(1, 1, -1),
            kernel.view(1, 1, -1),
            padding=radius,
        ).view(-1)

        pdf = pdf / (pdf.sum() * dx).clamp_min(self.eps)
        return pdf

    def _interp_pdf_1d(self, grid, pdf, query_row):
        """
        Linear interpolation of KDE grid values onto arbitrary query points
        """
        query_row = query_row.flatten()
        g = grid.numel()

        idx = torch.bucketize(query_row, grid)
        idx0 = torch.clamp(idx - 1, 0, g - 2)
        idx1 = idx0 + 1

        x0 = grid[idx0]
        x1 = grid[idx1]
        y0 = pdf[idx0]
        y1 = pdf[idx1]

        w = (query_row - x0) / (x1 - x0).clamp_min(self.eps)
        out = y0 + w * (y1 - y0)

        outside = (query_row < grid[0]) | (query_row > grid[-1])
        out[outside] = self.eps
        return out

    def _pdf_curve_1d(self, x_row):
        h = self._get_bandwidth(x_row)
        grid = self._make_grid_1d(x_row, h)
        pdf = self._kde_on_grid_1d(x_row, grid, h)
        return grid, pdf, h

    # ---------------------------
    # public api
    # ---------------------------
    def chunk_prob(self, x, x_point=None):
        """
        기존 이름 유지.
        의미는 'chunked exact KDE evaluation' 으로 정리.
        """
        x, x_squeezed = self._ensure_2d(x)

        if x_point is None:
            query_rows = [x[i] for i in range(x.shape[0])]
        else:
            query_rows = self._get_query_rows(x_point, x.shape[0], x.device, x.dtype)

        out = []
        for i in range(x.shape[0]):
            h = self._get_bandwidth(x[i])
            out.append(self._chunked_exact_prob_1d(x[i], query_rows[i], h))

        out = torch.stack(out, dim=0)
        if x_squeezed:
            out = out.squeeze(0)
        return out

    def prob(self, x, x_point=None):
        """
        Compatible interface:
          - prob(x): density at sample points x (fast path: grid + interpolation)
          - prob(x, x_point): density at arbitrary query points
              * exact chunked evaluation for small query sets
              * grid interpolation for large query sets
        """
        x, x_squeezed = self._ensure_2d(x)

        if x_point is None:
            query_rows = [x[i] for i in range(x.shape[0])]
            use_exact = False
        else:
            query_rows = self._get_query_rows(x_point, x.shape[0], x.device, x.dtype)
            use_exact = all(q.numel() <= self.exact_query_max_points for q in query_rows)

        out = []
        for i in range(x.shape[0]):
            if use_exact:
                h = self._get_bandwidth(x[i])
                pdf_i = self._chunked_exact_prob_1d(x[i], query_rows[i], h)
            else:
                grid, pdf_grid, _ = self._pdf_curve_1d(x[i])
                pdf_i = self._interp_pdf_1d(grid, pdf_grid, query_rows[i])

            out.append(pdf_i)

        out = torch.stack(out, dim=0)
        if x_squeezed:
            out = out.squeeze(0)
        return out

    def pdf_curve(self, x):
        """
        Plot 용 KDE pdf curve 반환
        Returns:
            grid: [G] or [B, G]
            pdf : [G] or [B, G]
        """
        x, x_squeezed = self._ensure_2d(x)

        grids = []
        pdfs = []
        for i in range(x.shape[0]):
            grid, pdf, _ = self._pdf_curve_1d(x[i])
            grids.append(grid)
            pdfs.append(pdf)

        grid = torch.stack(grids, dim=0)
        pdf = torch.stack(pdfs, dim=0)

        if x_squeezed:
            grid = grid.squeeze(0)
            pdf = pdf.squeeze(0)
        return grid, pdf

    def entropy(self, x):
        """
        Differential entropy of KDE-estimated distribution:
            H(p) = -∫ p(x) log p(x) dx
        """
        x, x_squeezed = self._ensure_2d(x)

        out = []
        for i in range(x.shape[0]):
            grid, pdf, _ = self._pdf_curve_1d(x[i])
            dx = grid[1] - grid[0]
            pdf = torch.clamp(pdf, min=self.eps)
            h = -(pdf * torch.log(pdf)).sum() * dx
            out.append(h)

        out = torch.stack(out)
        if x_squeezed:
            out = out.squeeze(0)
        return out

    def cross_entropy(self, p, q):
        """
        Cross entropy between KDE(p) and KDE(q):
            H(p, q) = -∫ p(x) log q(x) dx
        """
        p, p_squeezed = self._ensure_2d(p)
        q, _ = self._ensure_2d(q)

        if q.shape[0] == 1 and p.shape[0] > 1:
            q = q.expand(p.shape[0], -1)
        elif q.shape[0] != p.shape[0]:
            raise ValueError(f'Batch mismatch: p.shape={p.shape}, q.shape={q.shape}')

        out = []
        for i in range(p.shape[0]):
            hp = self._get_bandwidth(p[i])
            hq = self._get_bandwidth(q[i])

            pad = self.cut * torch.maximum(hp, hq)
            x_min = torch.minimum(torch.min(p[i]), torch.min(q[i])) - pad
            x_max = torch.maximum(torch.max(p[i]), torch.max(q[i])) + pad

            if torch.isclose(x_min, x_max):
                x_min = x_min - torch.maximum(hp, hq)
                x_max = x_max + torch.maximum(hp, hq)

            grid = torch.linspace(
                x_min,
                x_max,
                self.grid_size,
                device=p.device,
                dtype=p.dtype,
            )

            p_pdf = self._kde_on_grid_1d(p[i], grid, hp).clamp_min(self.eps)
            q_pdf = self._kde_on_grid_1d(q[i], grid, hq).clamp_min(self.eps)
            dx = grid[1] - grid[0]

            ce = -(p_pdf * torch.log(q_pdf)).sum() * dx
            out.append(ce)

        out = torch.stack(out)
        if p_squeezed:
            out = out.squeeze(0)
        return out

    def kl_divergence(self, p, q):
        ce = self.cross_entropy(p, q)
        self_entropy = self.entropy(p)
        return ce - self_entropy
    
    
class ShiftedLogKDE(nn.Module):
    """
    Support-constrained KDE for x > lower_bound.
    Internally:
        z = log(x - lower_bound)
    and do Gaussian KDE in z-space.
    """
    def __init__(self, lower_bound=2.0, eps=1e-12):
        super().__init__()
        self.base_kde = GaussianKernalDensityEstimation(chunk_size=8192, grid_size=16384)
        self.lower_bound = lower_bound
        self.eps = eps

    def _to_z(self, x):
        return torch.log((x - self.lower_bound).clamp_min(self.eps))

    def _to_x(self, z):
        return self.lower_bound + torch.exp(z)

    def prob(self, x, x_point=None):
        """
        x: samples from the target distribution, must satisfy x > lower_bound
        x_point: query points in x-space
        """
        if x_point is None:
            x_point = x

        z = self._to_z(x)

        # initialize output in x-space
        out = torch.zeros_like(x_point, dtype=z.dtype, device=z.device)

        valid = x_point > self.lower_bound
        if valid.any():
            z_point = self._to_z(x_point[valid])
            pz = self.base_kde.prob(z, z_point)
            out[valid] = pz / (x_point[valid] - self.lower_bound).clamp_min(self.eps)

        return out

    def pdf_curve(self, x):
        """
        Returns x_grid, pdf_x
        """
        z = self._to_z(x)
        z_grid, pdf_z = self.base_kde.pdf_curve(z)

        x_grid = self._to_x(z_grid)
        pdf_x = pdf_z / (x_grid - self.lower_bound).clamp_min(self.eps)

        # just for numerical safety
        pdf_x = pdf_x.clamp_min(0.0)
        return x_grid, pdf_x

    def entropy(self, x):
        """
        Numerical integration in x-space
        """
        x_grid, pdf_x = self.pdf_curve(x)
        dx = x_grid[1] - x_grid[0]
        pdf_x = pdf_x.clamp_min(self.eps)
        return -(pdf_x * torch.log(pdf_x)).sum() * dx