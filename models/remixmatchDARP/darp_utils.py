# ----
# modified from https://github.com/bbuing9/DARP/blob/master/common.py
# ----

import math
import torch
import numpy as np
from scipy import optimize


def estimate_pseudo(q_y, saved_q, num_class=10, alpha=2):
    q_y = q_y.cpu()
    saved_q = saved_q.cpu()
    pseudo_labels = torch.zeros(len(saved_q), num_class)
    k_probs = torch.zeros(num_class)
    for i in range(1, num_class + 1):
        i = num_class - i
        num_i = int(alpha * q_y[i])
        sorted_probs, idx = saved_q[:, i].sort(dim=0, descending=True)
        pseudo_labels[idx[:num_i], i] = 1
        k_probs[i] = sorted_probs[:num_i].sum()
    return pseudo_labels, (q_y + 1e-6) / (k_probs + 1e-6)


def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x / c)) - d


def safe_opt_solver(probs, target_distb, num_iter=10, th=0.1, num_newton=30):
    try:
        pseudo_refine = opt_solver(probs, target_distb, num_iter, th, num_newton)
    except:
        pseudo_refine = probs
    return pseudo_refine


def opt_solver(probs, target_distb, num_iter=10, th=0.1, num_newton=30):
    entropy = (-1 * probs * torch.log(probs + 1e-6)).sum(1)
    weights = 1 / entropy
    N, K = probs.size(0), probs.size(1)

    A, w, lam, nu, r, c = probs.cpu().numpy(), weights.cpu().numpy(), np.ones(N), np.ones(K), np.ones(N), target_distb.cpu().numpy()
    A_e = A / math.e
    X = np.exp(-1 * lam / w)
    Y = np.exp(-1 * nu.reshape(1, -1) / w.reshape(-1, 1))
    prev_Y = np.zeros(K)
    X_t, Y_t = X, Y
    for n in range(num_iter):
        # Normalization
        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom
        # Newton method
        Y_t = np.zeros(K)
        for i in range(K):
            Y_t[i] = optimize.newton(f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=th)
        prev_Y = Y_t
        Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))
    denom = np.sum(A_e * Y_t, 1)
    X_t = r / denom
    M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)
    return M
