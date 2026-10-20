import math
import torch
import torch.nn.functional as F

__all__ = ["log_hyp1f1_3half_1_stable", "rician_gsm_1d_logpdf", "fit_rician_gsm_1d_mle"]


def log_hyp1f1_3half_1_stable(z):
    """
    log 1F1(3/2; 1; z), z >= 0
    Uses:
        1F1(3/2;1;z)
        = exp(z) * [(1+z) i0e(z/2) + z i1e(z/2)]
    where i0e(t)=exp(-t)I0(t), i1e(t)=exp(-t)I1(t).
    """
    t = 0.5 * z
    b = (1.0 + z) * torch.special.i0e(t) + z * torch.special.i1e(t)
    return z + torch.log(b.clamp_min(torch.finfo(z.dtype).tiny))


def rician_gsm_1d_logpdf(y, rho, gamma, eps=1e-12):
    """
    1D marginal logpdf of Rician-style Gaussian precision mixture.

    Model:
        H = (rho + W) / sqrt(1 + rho^2), W ~ CN(0,1)
        G = |H|^2
        X | G=g ~ N(0, gamma^2 / g)

    y: arbitrary shape tensor
    rho: scalar tensor, rho >= 0
    gamma: scalar tensor, gamma > 0

    Note:
        This distribution has infinite variance for finite rho.
        gamma is a scale parameter, not a unit-variance normalizer.
    """
    y = y.reshape(-1)

    rho2 = rho ** 2
    a = 1.0 + rho2
    gamma2 = gamma ** 2

    A = a + (y ** 2) / (2.0 * gamma2)
    C = rho2 * a
    z = C / A

    log_const = (
        torch.log(a)
        - rho2
        - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=y.device, dtype=y.dtype) * gamma2)
        + torch.special.gammaln(torch.tensor(1.5, device=y.device, dtype=y.dtype))
    )

    log_kernel = -1.5 * torch.log(A) + log_hyp1f1_3half_1_stable(z)

    return log_const + log_kernel


def fit_rician_gsm_1d_mle(
    y,
    init_rho=1.0,
    init_gamma=None,
    max_iter=100,
    lr=0.5,
    tail_power=0.0,
    verbose=False,
):
    """
    Fit 2-parameter Rician-GSM 1D marginal by MLE.

    Parameters
    ----------
    y : torch.Tensor
        Samples, arbitrary shape.
    init_rho : float
        Initial rho >= 0.
    init_gamma : float or None
        Initial scale. If None, uses robust-ish std fallback.
    max_iter : int
        LBFGS max iterations.
    lr : float
        LBFGS learning rate.
    tail_power : float
        0.0 = plain MLE.
        >0 gives tail-weighted objective, like your student-t code.
    verbose : bool

    Returns
    -------
    rho_hat : scalar tensor
    gamma_hat : scalar tensor
    history : list of tuples
        [(loss, rho, gamma), ...]
    """
    y = y.reshape(-1).double()
    device = y.device
    dtype = y.dtype

    if init_gamma is None:
        # Since true variance may be infinite, std can be unstable.
        # MAD-style initialization is safer.
        med = y.median()
        mad = (y - med).abs().median()
        init_gamma = float((mad / 0.6745).clamp_min(1e-3).detach().cpu())

    init_rho_t = torch.tensor(float(init_rho), device=device, dtype=dtype).clamp_min(1e-6)
    init_gamma_t = torch.tensor(float(init_gamma), device=device, dtype=dtype).clamp_min(1e-6)

    # rho = softplus(alpha), gamma = exp(beta)
    init_alpha = torch.log(torch.expm1(init_rho_t))
    init_beta = torch.log(init_gamma_t)

    alpha = torch.nn.Parameter(init_alpha.clone())
    beta = torch.nn.Parameter(init_beta.clone())

    optimizer = torch.optim.LBFGS(
        [alpha, beta],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    if tail_power > 0:
        w = 1.0 + y.abs().pow(tail_power)
        w = w / w.mean()
    else:
        w = None

    history = []

    def closure():
        optimizer.zero_grad()

        rho = F.softplus(alpha)
        gamma = torch.exp(beta)

        logp = rician_gsm_1d_logpdf(y, rho, gamma)

        if w is None:
            loss = -logp.mean()
        else:
            loss = -(w * logp).mean()

        loss.backward()

        history.append((
            float(loss.detach().cpu()),
            float(rho.detach().cpu()),
            float(gamma.detach().cpu()),
        ))

        return loss

    optimizer.step(closure)

    rho_hat = F.softplus(alpha.detach())
    gamma_hat = torch.exp(beta.detach())

    if verbose:
        print("final rho   =", float(rho_hat.cpu()))
        print("final gamma =", float(gamma_hat.cpu()))
        print("final nll   =", history[-1][0])

    return rho_hat, gamma_hat, history