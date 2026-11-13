from math import sqrt

import torch


def solve_dual(g_of_mu, mu_min=1e-10, mu_max=1e10, steps=100):
    """Bisection for the Lagrange multiplier: g~(x*(mu)) decreases with mu."""
    with torch.no_grad():
        if g_of_mu(torch.tensor(0.0)) <= 0.0:
            return torch.tensor(0.0)
        lo, hi = torch.log10(torch.tensor(mu_min)), torch.log10(torch.tensor(mu_max))
        for _ in range(steps):
            mid = (lo + hi) / 2
            if g_of_mu(10.0**mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 10.0**hi


def mma_approximation(func, x_k, L, U, x_min, x_max):
    x = x_k.clone().requires_grad_()
    f_k = func(x)
    grad = torch.autograd.grad(f_k, x)[0]

    reg = 0.001 * grad.abs() + 1e-5 / (x_max - x_min)  # Svanberg 2002
    p = (U - x_k) ** 2 * (grad.clamp(min=0.0) + reg)
    q = (x_k - L) ** 2 * ((-grad).clamp(min=0.0) + reg)
    r = f_k.detach() - torch.sum(p / (U - x_k) + q / (x_k - L))

    def approx(x):
        return r + torch.sum(p / (U - x) + q / (x - L))

    return approx, p, q


def optimize_mma(f, g, x_0, x_min, x_max, n_iter=200, s=0.7, move=0.2, tol=1e-6):
    """Minimize f(x) subject to g(x) <= 0 and x_min <= x <= x_max."""
    span = x_max - x_min
    x = [x_0.clone()]
    L, U = [], []
    history = {"objective": [], "constraint": []}

    for k in range(n_iter):
        if k < 2:
            L.append(x[k] - s * span)
            U.append(x[k] + s * span)
        else:
            oscillating = (x[k] - x[k - 1]) * (x[k - 1] - x[k - 2]) < 0.0
            factor = torch.where(oscillating, torch.tensor(s), torch.tensor(1 / sqrt(s)))
            dist_L = (factor * (x[k - 1] - L[-1])).clamp(0.01 * span, 10 * span)
            dist_U = (factor * (U[-1] - x[k - 1])).clamp(0.01 * span, 10 * span)
            L.append(x[k] - dist_L)
            U.append(x[k] + dist_U)

        lower = torch.max(torch.max(x_min, 0.9 * L[k] + 0.1 * x[k]), x[k] - move * span)
        upper = torch.min(torch.min(x_max, 0.9 * U[k] + 0.1 * x[k]), x[k] + move * span)

        _, p_f, q_f = mma_approximation(f, x[k], L[k], U[k], x_min, x_max)
        g_approx, p_g, q_g = mma_approximation(g, x[k], L[k], U[k], x_min, x_max)
        history["objective"].append(float(f(x[k])))
        history["constraint"].append(float(g(x[k])))

        def x_star(mu, L=L[k], U=U[k], lower=lower, upper=upper):
            a = torch.sqrt(p_f + mu * p_g)
            b = torch.sqrt(q_f + mu * q_g)
            return torch.clamp((a * L + b * U) / (a + b), lower, upper)

        mu = solve_dual(lambda mu: g_approx(x_star(mu)))
        x.append(x_star(mu).detach())

        if torch.norm(x[-1] - x[-2]) < tol:
            break

    history["objective"].append(float(f(x[-1])))
    history["constraint"].append(float(g(x[-1])))
    history["x"] = torch.stack(x)
    return history
