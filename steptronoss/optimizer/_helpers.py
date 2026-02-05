from functools import partial

import torch
import torch.nn.functional as F

# https://arxiv.org/pdf/2505.16932
polar_express_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial )
polar_express_coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in polar_express_coeffs_list[:-1]] + [
    polar_express_coeffs_list[-1]
]


# https://arxiv.org/pdf/2505.16932, slower convergence (l=0.5)
polar_express_slow_convergence_coeffs_list = [
    (2.6708353888008127, -3.2853566190584504, 1.6385678673837945),
    (1.8756101482892078, -1.25067777555517, 0.3750677840207064),
    (1.8750001724801866, -1.2500003448894914, 0.37500017240930517),
    (1.875, -1.25, 0.375),
]
# safety factor for numerical stability (but exclude last polynomial )
polar_express_slow_convergence_coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in polar_express_slow_convergence_coeffs_list[:-1]
] + [polar_express_slow_convergence_coeffs_list[-1]]


ZEROPOWER_COEFFS = {
    "default": (3.4445, -4.7750, 2.0315),
    "youjc_250416": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "youjc_6": [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ],
    "sujl_6": [
        (4140 / 1024, -7553 / 1024, 3571 / 1024),
        (3892 / 1024, -6637 / 1024, 2973 / 1024),
        (3668 / 1024, -6456 / 1024, 3021 / 1024),
        (3248 / 1024, -6211 / 1024, 3292 / 1024),
        (2792 / 1024, -5759 / 1024, 3796 / 1024),
        (3176 / 1024, -5507 / 1024, 4048 / 1024),
    ],
    "polar_express": polar_express_coeffs_list,
    "polar_express_slow_convergence": polar_express_slow_convergence_coeffs_list,
}


def zeropower(G, steps, run_ns_in_fp16, coeffs):
    # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    assert G.ndim >= 2

    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    if run_ns_in_fp16:
        X = X.to(torch.float16)

    if isinstance(coeffs[0], float):
        coeffs = [coeffs]
    if len(coeffs) < steps:
        coeffs += [coeffs[-1]] * (steps - len(coeffs))
    elif len(coeffs) > steps:
        coeffs = coeffs[:steps]

    for a, b, c in coeffs:
        A = X @ X.mT
        if X.dim() == 3:
            B = torch.baddbmm(A, A, A, alpha=c, beta=b)
            X = torch.baddbmm(X, B, X, beta=a)
        elif X.dim() == 2:
            B = torch.addmm(A, A, A, alpha=c, beta=b)
            X = torch.addmm(X, B, X, beta=a)

    if G.size(-2) > G.size(-1):
        X = X.mT.contiguous()

    return X


def zeropower_via_svd(G, steps):
    if G.ndim > 2:
        shape = G.shape
        G = G.view(-1, shape[-2], shape[-1])
        out = []
        for g in G.unbind(0):
            U, S, V = g.float().svd()
            out.append(U @ V.T)
        return torch.stack(out, dim=0).view(shape)
    else:
        U, S, V = G.float().svd()
        return U @ V.T


def double_normalize(G, steps):
    order = [-2, -1] if G.shape[-2] < G.shape[-1] else [-1, -2]
    for _ in range(steps):
        for dim in order:
            G = G / (G.norm(dim=dim, keepdim=True, p=2) + 1e-7)
    return G


def sinkhorn_knopp(G, steps):
    order = [-2, -1] if G.shape[-2] < G.shape[-1] else [-1, -2]
    for _ in range(steps):
        for dim in order:
            G = G / (G.norm(dim=dim, keepdim=True, p=1) + 1e-7)
    return G * G.shape[order[-1]] ** 0.5


def dualize_1_to_p(G, steps=None, p=1):
    # http://arxiv.org/abs/2409.20325 page 19 proposition 8
    return G / (torch.linalg.vector_norm(G, dim=1, keepdim=True, ord=p) + 1e-7) * G.shape[1] ** 0.5


def dualize_p_to_infinity(G, steps=None, p=1):
    # http://arxiv.org/abs/2409.20325 page 19 proposition 8
    p = 1 if p == "inf" else p / (p - 1)  # dual norm
    return G / (torch.linalg.vector_norm(G, dim=2, keepdim=True, ord=p) + 1e-7) * (G.shape[2] / G.shape[1] ** 0.5)


def dualize_schatten_2_norm(G, steps=None):
    return (
        G
        / (torch.linalg.matrix_norm(G, ord="fro", dim=(-2, -1), keepdim=True) + 1e-7)
        * min(G.shape[-2], G.shape[-1]) ** 0.5
    )


def get_zeropower_fn(fn_name, run_ns_in_fp16=True):
    if fn_name == "svd":
        return zeropower_via_svd
    elif fn_name == "double_normalize":
        # not an orthogonalization function
        return double_normalize
    elif fn_name == "sinkhorn_knopp":
        # not an orthogonalization function
        return sinkhorn_knopp
    elif fn_name == "dualize_1_to_1":
        # not an orthogonalization function
        return partial(dualize_1_to_p, p=1)
    elif fn_name == "dualize_inf_to_inf":
        # not an orthogonalization function
        return partial(dualize_p_to_infinity, p=float("inf"))
    elif fn_name == "dualize_schatten_2_norm":
        # not an orthogonalization function
        return dualize_schatten_2_norm
    else:
        return partial(
            zeropower,
            run_ns_in_fp16=run_ns_in_fp16,
            coeffs=ZEROPOWER_COEFFS[fn_name],
        )


def get_step_fun(fn, threshold, steps=5):
    # https://x.com/YouJiacheng/status/1930988035195478303

    # @torch.compile
    @torch.no_grad()
    def step_fun(G):
        uv = fn(G, steps)
        uv2 = fn(G - threshold * uv, steps)
        return (uv + uv2) / 2

    return step_fun


def get_clip_fun(fn, threshold, steps=5):

    # @torch.compile
    @torch.no_grad()
    def clip_fun(M):

        uv = fn(M, steps)
        uv2 = fn(M - threshold * uv, steps)
        x = (threshold * torch.eye(M.shape[-2], device=M.device, dtype=M.dtype) - M @ uv.mT) @ uv2
        return (M + threshold * uv + x) / 2

    return clip_fun


def get_max_singular_value(M):
    return torch.linalg.svdvals(M.float()).max()


# @torch.compile
@torch.no_grad()
def weight_decay_in_spectral_norm(M, steps=10, pre_iters=3):
    # if M.ndim == 3:
    #     grad_spectral_norm_refs = []
    #     for p_item in M.unbind(0):
    #         u_true, sigma, v_true = torch.linalg.svd(p_item.data.float(), full_matrices=False)
    #         grad_spectral_norm_refs.append(u_true[:, 0, None] @ v_true[None, :, 0])
    #     grad_spectral_norm_ref = torch.stack(grad_spectral_norm_refs, dim=0).to(M.dtype)
    # elif M.ndim == 2:
    #     u_true, sigma, v_true = torch.linalg.svd(M.data.float(), full_matrices=False)
    #     grad_spectral_norm_ref = (u_true[:, 0, None] @ v_true[None, :, 0]).to(M.dtype)

    should_transpose = M.shape[-1] > M.shape[-2]
    if should_transpose:  # NOTE here we use tall matrix
        M = M.mT

    # make M to have a slightly less than 1 spectral norm
    M /= (M.norm(dim=(-2, -1), keepdim=True) + 1e-7) / M.shape[-1] ** 0.5

    MTM = M.mT @ M
    for _ in range(pre_iters):
        M = M @ MTM
    v = torch.randn((M.shape[0], M.shape[-1], 1), device=M.device, dtype=M.dtype)
    MTM = M.mT @ M

    for _ in range(steps):
        v = MTM @ v
        v = F.normalize(v, dim=1)

    v1 = v.squeeze(-1)
    u1 = F.normalize(M @ v, dim=1).squeeze(-1)
    grad = u1.unsqueeze(-1) @ v1.unsqueeze(1)

    # u1_true = u_true[:, 0]
    # v1_true = v_true[0, :]
    # print(torch.abs(torch.dot(u1[0], v1_true)))
    # print(torch.abs(torch.dot(v1[0], u1_true)))
    if should_transpose:
        grad = grad.mT.contiguous()
    # print(should_transpose, grad.shape)
    return grad
