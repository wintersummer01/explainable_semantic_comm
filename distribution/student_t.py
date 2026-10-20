
import torch
import math
import torch.nn.functional as F

__all__ = ["student_t_logpdf_unitvar", "fit_student_t_nu_mle", "fit_student_t_nu_profile"]


def student_t_logpdf_unitvar(y, nu):
    """
    y: arbitrary shape
    nu: scalar tensor, nu > 2
    unit-variance Student-t
    """
    c = (
        torch.special.gammaln((nu + 1.0) / 2.0)
        - torch.special.gammaln(nu / 2.0)
        - 0.5 * torch.log(torch.tensor(math.pi, device=y.device, dtype=y.dtype) * (nu - 2.0))
    )
    log_kernel = -0.5 * (nu + 1.0) * torch.log1p((y ** 2) / (nu - 2.0))
    return c + log_kernel


def fit_student_t_nu_mle(y, init_nu=10.0, max_iter=100, lr=0.5, verbose=False):
    """
    y: tensor of samples, e.g. [N] or [B, N]
    return: nu_hat (scalar tensor)
    """
    y = y .reshape(-1)

    # nu = 2 + softplus(alpha)
    init_alpha = torch.log(torch.exp(torch.tensor(init_nu - 2.0, device=y.device, dtype=y.dtype)) - 1.0)
    alpha = torch.nn.Parameter(init_alpha.clone())

    optimizer = torch.optim.LBFGS(
        [alpha],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    history = []

    def closure():
        optimizer.zero_grad()
        nu = 2.0 + F.softplus(alpha)
        loss = -student_t_logpdf_unitvar(y, nu).mean()
        loss.backward()
        history.append((float(loss.detach().cpu()), float(nu.detach().cpu())))
        return loss

    optimizer.step(closure)

    nu_hat = 2.0 + F.softplus(alpha.detach())
    if verbose:
        print("final nu =", float(nu_hat.cpu()))
        print("final nll =", history[-1][0])

    return nu_hat, history

def fit_student_t_nu_profile(
    y,
    nu_min=2.05,
    nu_max=100.0,
    coarse_steps=200,
    refine_steps=200,
    tail_power=0.0,   # 0이면 plain MLE, >0이면 tail-weighted fit
):
    """
    y: [N] or arbitrary shape
    returns:
        nu_hat, history
    """
    y = y.reshape(-1).double()

    # if tail_power > 0:
    #     w = 1.0 + y.abs().pow(tail_power)
    #     w = w / w.mean()
    # else:
    #     w = None
    w = 1.0 + y.abs().pow(tail_power)
    w = w / w.mean()

    def nll(nu):
        logp = student_t_logpdf_unitvar(y, nu)
        if w is None:
            return -logp.mean()
        else:
            return -(w * logp).mean()

    def search(lo, hi, steps):
        # log(nu-2) space에서 search: nu≈2 근처 해상도 확보
        eta_lo = math.log(lo - 2.0)
        eta_hi = math.log(hi - 2.0)
        etas = torch.linspace(eta_lo, eta_hi, steps, device=y.device, dtype=y.dtype)
        nus = 2.0 + torch.exp(etas)

        vals = torch.stack([nll(nu) for nu in nus])
        idx = torch.argmin(vals)
        return nus, vals, idx

    # coarse
    nus1, vals1, idx1 = search(nu_min, nu_max, coarse_steps)

    left = nus1[max(idx1 - 1, 0)].item()
    right = nus1[min(idx1 + 1, coarse_steps - 1)].item()

    # refine
    nus2, vals2, idx2 = search(left, right, refine_steps)

    nu_hat = nus2[idx2]
    history = {
        "coarse_nus": nus1.detach().cpu(),
        "coarse_nll": vals1.detach().cpu(),
        "refine_nus": nus2.detach().cpu(),
        "refine_nll": vals2.detach().cpu(),
        "final_nll": vals2[idx2].detach().cpu()
    }
    return nu_hat, history