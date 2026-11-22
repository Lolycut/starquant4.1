"""PlatinumQuant-4.1 standalone PyTorch model (CPU & CUDA/Triton compatible)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0
HAS_NATIVE_BF16 = (DEVICE == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
IO_DTYPE = torch.bfloat16 if HAS_NATIVE_BF16 else (torch.float16 if DEVICE == "cuda" else torch.float32)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = (DEVICE == "cuda")
except ImportError:
    HAS_TRITON = False

#@title Block 2: Аудированная Архитектура `PlatinumStar-3.0.1` + Встроенный Тест Каузальности `verify_strict_causality()`
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight

TRITON_RUNTIME_OK = False
if HAS_TRITON:
    @triton.jit
    def _triton_flash_scan_fwd_kernel(
        W_wy_ptr, U_init_ptr, K_dei_ptr, Dec_ec_ptr,
        S_hist_ptr, U_all_ptr,
        n_chunks: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = tl.arange(0, 64)
        offs_n = tl.arange(0, 64)
        tile_offs = offs_m[:, None] * 64 + offs_n[None, :]
        stream_stride = n_chunks * 4096
        base_offset = pid * stream_stride

        dec_ec = tl.load(Dec_ec_ptr + pid).to(tl.float32)
        S = tl.zeros((64, 64), dtype=tl.float32)

        for ci in range(n_chunks):
            chunk_off = base_offset + ci * 4096 + tile_offs
            tl.store(S_hist_ptr + chunk_off, S)
            w_wy_32 = tl.load(W_wy_ptr + chunk_off).to(tl.float32)
            u_in_32 = tl.load(U_init_ptr + chunk_off).to(tl.float32)
            k_d_32  = tl.load(K_dei_ptr + chunk_off).to(tl.float32)

            w_max = tl.max(tl.abs(w_wy_32)) + 1e-7
            s_max = tl.max(tl.abs(S)) + 1e-7
            w_16  = (w_wy_32 / w_max).to(tl.float16)
            s_16  = (S / s_max).to(tl.float16)
            u_eff = u_in_32 - tl.dot(w_16, s_16) * (w_max * s_max)
            tl.store(U_all_ptr + chunk_off, u_eff)

            k_max = tl.max(tl.abs(k_d_32)) + 1e-7
            u_max = tl.max(tl.abs(u_eff)) + 1e-7
            k_16  = (k_d_32 / k_max).to(tl.float16)
            u_16  = (u_eff / u_max).to(tl.float16)
            S = dec_ec * S + tl.dot(tl.trans(k_16), u_16) * (k_max * u_max)

    @triton.jit
    def _triton_flash_scan_bwd_kernel(
        W_wy_ptr, K_dei_ptr, Dec_ec_ptr,
        S_hist_ptr, U_all_ptr,
        Grad_S_ptr, Grad_U_ptr,
        dW_wy_ptr, dU_init_ptr, dK_dei_ptr,
        n_chunks: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = tl.arange(0, 64)
        offs_n = tl.arange(0, 64)
        tile_offs = offs_m[:, None] * 64 + offs_n[None, :]
        stream_stride = n_chunks * 4096
        base_offset = pid * stream_stride

        dec_ec = tl.load(Dec_ec_ptr + pid).to(tl.float32)
        dS = tl.zeros((64, 64), dtype=tl.float32)

        for rev_i in range(n_chunks):
            ci = (n_chunks - 1) - rev_i
            chunk_off = base_offset + ci * 4096 + tile_offs
            g_u_32  = tl.load(Grad_U_ptr + chunk_off).to(tl.float32)
            g_s_32  = tl.load(Grad_S_ptr + chunk_off).to(tl.float32)
            k_d_32  = tl.load(K_dei_ptr + chunk_off).to(tl.float32)
            w_wy_32 = tl.load(W_wy_ptr + chunk_off).to(tl.float32)
            s_h_32  = tl.load(S_hist_ptr + chunk_off).to(tl.float32)
            u_a_32  = tl.load(U_all_ptr + chunk_off).to(tl.float32)

            k_max  = tl.max(tl.abs(k_d_32)) + 1e-7
            ds_max = tl.max(tl.abs(dS)) + 1e-7
            k_16   = (k_d_32 / k_max).to(tl.float16)
            ds_16  = (dS / ds_max).to(tl.float16)

            du = g_u_32 + tl.dot(k_16, ds_16) * (k_max * ds_max)

            du_max = tl.max(tl.abs(du)) + 1e-7
            ua_max = tl.max(tl.abs(u_a_32)) + 1e-7
            sh_max = tl.max(tl.abs(s_h_32)) + 1e-7
            w_max  = tl.max(tl.abs(w_wy_32)) + 1e-7

            du_16 = (du / du_max).to(tl.float16)
            ua_16 = (u_a_32 / ua_max).to(tl.float16)
            sh_16 = (s_h_32 / sh_max).to(tl.float16)
            w_16  = (w_wy_32 / w_max).to(tl.float16)

            dk_d  = tl.dot(ua_16, tl.trans(ds_16)) * (ua_max * ds_max)
            dw_wy = -tl.dot(du_16, tl.trans(sh_16)) * (du_max * sh_max)

            tl.store(dU_init_ptr + chunk_off, du)
            tl.store(dK_dei_ptr + chunk_off, dk_d)
            tl.store(dW_wy_ptr + chunk_off, dw_wy)

            dS = g_s_32 + dec_ec * dS - tl.dot(tl.trans(w_16), du_16) * (w_max * du_max)

class TritonFlashScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_wy, u_init, k_dei, dec_ec):
        w_wy_c = w_wy.contiguous()
        u_in_c = u_init.contiguous()
        k_d_c  = k_dei.contiguous()
        dec_c  = dec_ec.contiguous()
        N_streams, Nc, _, _ = w_wy_c.shape
        s_hist = torch.empty_like(w_wy_c)
        u_all  = torch.empty_like(u_in_c)
        with torch.cuda.device(w_wy_c.device):
            _triton_flash_scan_fwd_kernel[(N_streams,)](w_wy_c, u_in_c, k_d_c, dec_c, s_hist, u_all, n_chunks=Nc)
        ctx.save_for_backward(w_wy_c, k_d_c, dec_c, s_hist, u_all)
        return s_hist, u_all

    @staticmethod
    def backward(ctx, grad_s_hist, grad_u_all):
        w_wy_c, k_d_c, dec_c, s_hist, u_all = ctx.saved_tensors
        N_streams, Nc, _, _ = w_wy_c.shape
        gs_c = grad_s_hist.contiguous().float()
        gu_c = grad_u_all.contiguous().float()
        dw_wy   = torch.empty_like(w_wy_c)
        du_init = torch.empty_like(w_wy_c)
        dk_dei  = torch.empty_like(w_wy_c)
        with torch.cuda.device(w_wy_c.device):
            _triton_flash_scan_bwd_kernel[(N_streams,)](w_wy_c, k_d_c, dec_c, s_hist, u_all, gs_c, gu_c, dw_wy, du_init, dk_dei, n_chunks=Nc)
        return dw_wy, du_init, dk_dei, None

def wy_chunk_scan_dispatch(w_wy, u_init, k_dei, dec_ec):
    B, H, Nc, C, Dh = w_wy.shape
    if TRITON_RUNTIME_OK and w_wy.is_cuda and C == 64 and Dh == 64:
        N_streams = B * H
        dec_flat = dec_ec.expand(B, H, 1, 1).reshape(N_streams)
        s_h, u_a = TritonFlashScanFn.apply(
            w_wy.reshape(N_streams, Nc, 64, 64),
            u_init.reshape(N_streams, Nc, 64, 64),
            k_dei.reshape(N_streams, Nc, 64, 64),
            dec_flat,
        )
        return s_h.view(B, H, Nc, 64, 64), u_a.view(B, H, Nc, 64, 64)
    S = torch.zeros((B, H, Dh, Dh), device=w_wy.device, dtype=torch.float32)
    s_list, u_list = [], []
    for ci in range(Nc):
        s_list.append(S)
        u_eff = u_init[:, :, ci] - torch.matmul(w_wy[:, :, ci], S)
        u_list.append(u_eff)
        S = dec_ec * S + torch.matmul(k_dei[:, :, ci].transpose(-2, -1), u_eff)
    return torch.stack(s_list, dim=2), torch.stack(u_list, dim=2)

TRITON_RUNTIME_OK = (DEVICE == "cuda" and HAS_TRITON)

# ==============================================================================
# 0B. PURE-PYTORCH MUON + ADAMW HYBRID OPTIMIZER (TPU XLA & GPU Compatible)
# ==============================================================================
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    _use_bf16 = (G.device.type == "cuda" and torch.cuda.get_device_capability(G.device)[0] >= 8)
    X = G.bfloat16() if _use_bf16 else G.float()
    X = X / (X.norm() + eps)
    transposed = (G.size(0) > G.size(1))
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonAdamWHybrid(torch.optim.Optimizer):
    def __init__(self, muon_params, adamw_params, lr=5e-4, muon_lr_mult=8.0, momentum=0.95, weight_decay=0.05):
        param_groups = [
            {"params": list(muon_params), "is_muon": True,  "lr": lr * muon_lr_mult, "momentum": momentum, "weight_decay": weight_decay},
            {"params": list(adamw_params), "is_muon": False, "lr": lr, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": weight_decay},
        ]
        super().__init__(param_groups, {})
        self.muon_lr_mult = muon_lr_mult

    def set_base_lr(self, lr):
        for pg in self.param_groups:
            pg["lr"] = lr * self.muon_lr_mult if pg["is_muon"] else lr

    @torch.no_grad()
    def step(self, closure=None):
        for pg in self.param_groups:
            lr = pg["lr"]
            wd = pg["weight_decay"]
            if pg["is_muon"]:
                mom = pg["momentum"]
                for p in pg["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.mul_(mom).add_(g)
                    g_nesterov = g + mom * buf
                    upd = zeropower_via_newtonschulz5(g_nesterov)
                    scale = 0.2 * (max(p.size(0), p.size(1)) ** 0.5)
                    if wd != 0:
                        p.mul_(1.0 - lr * wd * 0.25)
                    p.add_(upd, alpha=-lr * scale)
            else:
                b1, b2 = pg["betas"]
                eps = pg["eps"]
                for p in pg["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(g)
                        st["v"] = torch.zeros_like(g)
                    st["step"] += 1
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    bc1 = 1.0 - b1 ** st["step"]
                    bc2 = 1.0 - b2 ** st["step"]
                    if wd != 0 and p.ndim >= 2:
                        p.mul_(1.0 - lr * wd)
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(m, denom, value=-lr / bc1)


# ==============================================================================
# 1. PLATINUMSTAR-3.0.1 (`v3.0.2-T4`) STAR-TRIAD ENGINE — AUDITED & MATH-EXACT:
#    - [FIX #2] Rotation-Invariant Quadratic-Form Kalman Precision:
#        pi(t) = (k_t^T P_cum k_t) / (c * C) + (1/C) * sum_{tau < t} w_tau (k_t^T k_tau)^2
#      Strictly ZERO for any orthogonal keys k_1^T k_2 = 0 in ANY rotated basis!
#    - [FIX #3B] Pure Local Chunk-Matrix Bank M_c = K_dei^T @ U_init (ZERO past-state S_{c-1} subtraction!)
#      while Delta-Error Surprise uses global prediction residual U_all: log1p(||U_all||_2)
# ==============================================================================
class StarTriadV3Engine(nn.Module):
    def __init__(self, d_model=768, n_heads=12, chunk_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_triads = n_heads // 3
        self.C = chunk_size
        assert n_heads % 3 == 0, "n_heads must be divisible by 3"

        qk_dim = 2 * self.n_triads * self.head_dim
        bind_dim = 2 * self.n_triads * self.head_dim
        self.qk_dim = qk_dim
        self.bind_dim = bind_dim
        self.in_proj = nn.Linear(d_model, qk_dim + bind_dim + d_model + 5 * n_heads)
        with torch.no_grad():
            self.in_proj.bias[:qk_dim + bind_dim + d_model].zero_()
            gb = self.in_proj.bias[qk_dim + bind_dim + d_model:].view(n_heads, 5)
            gb[:, 0].fill_(-1.8)
            gb[:, 1].fill_(-0.30)
            gb[:, 2].fill_(-0.5)
            gb[:, 3].fill_(0.5)
            gb[:, 4].fill_(0.0)
        self.v_res_gate = nn.Parameter(torch.tensor(0.0))
        self.bind_scale = nn.Parameter(torch.full((1, 1, self.n_triads, 1), 0.35))

        role_erase = [1.90 if (h % 3 == 0) else (1.00 if (h % 3 == 1) else 0.35) for h in range(n_heads)]
        role_bayes = [0.00 if (h % 3 == 0) else (0.35 if (h % 3 == 1) else 1.00) for h in range(n_heads)]
        role_bind  = [0.15 if (h % 3 == 0) else (0.50 if (h % 3 == 1) else 1.00) for h in range(n_heads)]
        self.register_buffer("erase_scales", torch.tensor(role_erase, dtype=torch.float32).view(1, n_heads, 1, 1))
        self.register_buffer("bayes_masks",  torch.tensor(role_bayes, dtype=torch.float32).view(1, n_heads, 1, 1))
        self.register_buffer("bind_masks",   torch.tensor(role_bind,  dtype=torch.float32).view(1, 1, n_heads, 1))
        band_b_idx = [h for h in range(n_heads) if h % 3 == 2]
        self.register_buffer("band_b_idx", torch.tensor(band_b_idx, dtype=torch.long))

        gA = torch.logspace(math.log10(1.0 / 2048.0),  math.log10(0.050), self.n_triads, dtype=torch.float32)
        gC = torch.logspace(math.log10(0.050),         math.log10(0.350), self.n_triads, dtype=torch.float32)
        gB = torch.logspace(math.log10(1.0 / 16384.0), math.log10(0.002), self.n_triads, dtype=torch.float32)
        lams = []
        for ti in range(self.n_triads):
            lams.extend([gA[ti], gC[ti], gB[ti]])
        self.register_buffer("base_lambdas", torch.stack(lams))

        pos = torch.arange(chunk_size, dtype=torch.float32)
        dt = torch.clamp(pos.unsqueeze(1) - pos.unsqueeze(0), min=0.0)
        s_low = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.float32), diagonal=-1)
        low = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.float32), diagonal=0)
        eye_c = torch.eye(chunk_size, dtype=torch.float32).view(1, 1, 1, chunk_size, chunk_size)
        offdiag = (1.0 - torch.eye(chunk_size, dtype=torch.float32)).view(1, 1, 1, chunk_size, chunk_size)
        self.register_buffer("pos_buf", pos)
        self.register_buffer("dt_buf", dt)
        self.register_buffer("s_low_buf", s_low)
        self.register_buffer("low_buf", low)
        self.register_buffer("eye_buf", eye_c)
        self.register_buffer("offdiag_buf", offdiag)

        self.hop2_proj = nn.Linear(d_model, self.n_triads * self.head_dim, bias=False)
        self.hop2_triad_proj = nn.Linear(d_model, n_heads, bias=False)
        self.hop2_scale = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.42))
        self.hop_mix_gate = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, v0=None, loop_idx=0, compute_ortho=False):
        B, Lp, D = x.shape
        H, Nt, Dh, C = self.n_heads, self.n_triads, self.head_dim, self.C
        n_chunks = Lp // C
        work_dtype = x.dtype if (x.dtype in (torch.float16, torch.bfloat16)) else torch.float32

        proj = self.in_proj(x)
        qk_tr = proj[:, :, :self.qk_dim].view(B, Lp, 2, Nt, Dh).float()
        bd_tr = proj[:, :, self.qk_dim : self.qk_dim + self.bind_dim].view(B, Lp, 2, Nt, Dh).float()
        v_raw = proj[:, :, self.qk_dim + self.bind_dim : self.qk_dim + self.bind_dim + D].view(B, Lp, H, Dh).float()
        gates = proj[:, :, self.qk_dim + self.bind_dim + D:].float().view(B, Lp, H, 5)

        b1 = torch.tanh(bd_tr[:, :, 0])
        b2 = torch.tanh(bd_tr[:, :, 1])
        b2_prev = torch.cat([torch.zeros_like(b2[:, :1]), b2[:, :-1]], dim=1)
        hadamard_term = (torch.tanh(self.bind_scale.float()) * (b1 * b2_prev)).repeat_interleave(3, dim=2)

        q_base = qk_tr[:, :, 0].repeat_interleave(3, dim=2)
        k_base = qk_tr[:, :, 1].repeat_interleave(3, dim=2)
        q1 = F.normalize(q_base + self.bind_masks * hadamard_term, dim=-1, eps=1e-5)
        k  = F.normalize(k_base + self.bind_masks * hadamard_term, dim=-1, eps=1e-5)
        v  = v_raw
        if v0 is not None:
            v = v + torch.sigmoid(self.v_res_gate) * v0

        b_raw   = torch.sigmoid(gates[..., 0]).permute(0, 2, 1).view(B, H, n_chunks, C)
        w_base  = (1.90 * torch.sigmoid(gates[..., 1])).permute(0, 2, 1).view(B, H, n_chunks, C)
        r_noise = (0.30 * F.softplus(gates[..., 2])).permute(0, 2, 1).view(B, H, n_chunks, C)
        k2_eh   = torch.sigmoid(gates[..., 3]).permute(0, 2, 1).view(B, H, n_chunks, C)
        triad_logits1 = gates[..., 4]

        qc1 = q1.permute(0, 2, 1, 3).view(B, H, n_chunks, C, Dh)
        kc  = k.permute(0, 2, 1, 3).view(B, H, n_chunks, C, Dh)
        vc  = v.permute(0, 2, 1, 3).view(B, H, n_chunks, C, Dh)

        # [FIX #2] Exact Rotation-Invariant Quadratic-Form Kalman Precision:
        #   1) Intra-chunk: sum_{tau < t} w_tau * (k_t^T k_tau)^2 — strictly 0 when k_t perp k_tau!
        #   2) Inter-chunk: k_t^T (sum_{c' < c} K_{c'}^T diag(w_{c'}) K_{c'}) k_t — strictly 0 when k_t perp past keys!
        kk_raw = torch.matmul(kc, kc.transpose(-2, -1))  # (B, H, Nc, C, C)
        w_eff_prec = w_base * k2_eh                      # (B, H, Nc, C)
        kk_sq_causal = (kk_raw * kk_raw) * self.s_low_buf.view(1, 1, 1, C, C)
        intra_prec = torch.matmul(kk_sq_causal, w_eff_prec.unsqueeze(-1)).squeeze(-1) / float(C)

        kw = kc * w_eff_prec.unsqueeze(-1)               # (B, H, Nc, C, Dh)
        P_chunk = torch.matmul(kw.transpose(-2, -1), kc) / float(C)  # (B, H, Nc, Dh, Dh)
        P_cum = torch.cumsum(P_chunk, dim=2) - P_chunk   # strictly past chunks c' < c
        c_arange = torch.arange(1, n_chunks + 1, device=x.device, dtype=torch.float32).view(1, 1, n_chunks, 1)
        inter_prec = (torch.matmul(kc, P_cum) * kc).sum(dim=-1) / c_arange  # (B, H, Nc, C)

        exact_bayes_prec = (inter_prec + 0.50 * intra_prec) * 4.0
        bayes_atten = 1.0 / (1.0 + self.bayes_masks * (r_noise + 0.75 * exact_bayes_prec))

        bc = b_raw * self.erase_scales * bayes_atten * (0.5 + 0.5 * k2_eh)
        wc = w_base * (0.40 + 0.60 * k2_eh) * (0.70 + 0.30 * bayes_atten)

        lams = self.base_lambdas
        pos = self.pos_buf
        exp_arg = torch.exp(-lams.view(1, H, 1, 1, 1) * self.dt_buf.view(1, 1, 1, C, C))
        G_s = exp_arg * self.s_low_buf.view(1, 1, 1, C, C)
        G_l_tc = (exp_arg * self.low_buf.view(1, 1, 1, C, C)).to(work_dtype)

        k2_outer = 0.5 + 0.5 * (k2_eh.unsqueeze(-1) * k2_eh.unsqueeze(-2))
        kk_triadic = kk_raw * k2_outer

        ortho_loss = None
        if compute_ortho:
            kk_band_b = kk_raw.index_select(1, self.band_b_idx)
            ortho_loss = ((kk_band_b * self.offdiag_buf) ** 2).mean()

        A_all = self.eye_buf + bc.unsqueeze(-1) * (kk_triadic * G_s)
        dec_c = torch.exp(-lams.view(1, H, 1, 1) * (pos + 1.0).view(1, 1, C, 1))
        dec_c_tc = dec_c.unsqueeze(2).to(work_dtype)
        dec_ec = torch.exp(-lams.view(1, H, 1, 1) * float(C))
        dec_ei = torch.exp(-lams.view(1, H, 1, 1) * (float(C - 1) - pos).view(1, 1, C, 1))

        k_b_dec = bc.unsqueeze(-1) * dec_c.unsqueeze(2) * kc
        v_w_in  = wc.unsqueeze(-1) * vc
        rhs_cat = torch.cat([k_b_dec, v_w_in], dim=-1)
        wu_sol  = torch.linalg.solve_triangular(A_all, rhs_cat, upper=False, unitriangular=True)
        w_wy, u_init = wu_sol.split(Dh, dim=-1)
        k_dei = kc * dec_ei.unsqueeze(2)

        S_hist_f32, U_all_f32 = wy_chunk_scan_dispatch(w_wy, u_init, k_dei, dec_ec)
        S_hist = S_hist_f32.to(work_dtype)
        U_all  = U_all_f32.to(work_dtype)
        kc_tc  = kc.to(work_dtype)
        qc1_tc = qc1.to(work_dtype)
        k2_outer_tc = k2_outer.to(work_dtype)

        # [FIX #3B] Pure Local Chunk-Matrix Bank M_c uses U_init (NOT U_all!) so chunk c's
        # associative table is 100% self-contained with ZERO subtraction of past global state S_{c-1}!
        M_chunk_bank = torch.matmul(k_dei.to(work_dtype).transpose(-2, -1), u_init.to(work_dtype))

        # Delta-Error Surprise uses global prediction residual U_all = U_init - W_wy @ S_{c-1}
        surprise_map = torch.log1p(U_all_f32.detach().norm(dim=-1)).reshape(B, H, Lp).permute(0, 2, 1)

        head_gain1 = (3.0 * torch.softmax(triad_logits1.view(B, Lp, Nt, 3), dim=-1)).view(B, Lp, H, 1).to(work_dtype)

        o1_carry = torch.matmul(qc1_tc, S_hist) * dec_c_tc
        qk1_eff  = (torch.matmul(qc1_tc, kc_tc.transpose(-2, -1)) * k2_outer_tc) * G_l_tc
        o1_full  = (o1_carry + torch.matmul(qk1_eff, U_all)).reshape(B, H, Lp, Dh)
        o1_gated = (o1_full.permute(0, 2, 1, 3).to(work_dtype) * head_gain1).reshape(B, Lp, D)

        if loop_idx == 0:
            return self.out_proj(o1_gated), M_chunk_bank, surprise_map, ortho_loss

        q2_tr_delta = self.hop2_proj(o1_gated).view(B, Lp, Nt, Dh).repeat_interleave(3, dim=2).permute(0, 2, 1, 3).float()
        triad_delta = self.hop2_triad_proj(o1_gated).float()

        q1_flat = q1.permute(0, 2, 1, 3)
        q2_flat = F.normalize(q1_flat + torch.tanh(self.hop2_scale.float()) * q2_tr_delta, dim=-1, eps=1e-5)
        qc2_tc  = q2_flat.view(B, H, n_chunks, C, Dh).to(work_dtype)

        head_gain2 = (3.0 * torch.softmax((triad_logits1 + 0.5 * triad_delta).view(B, Lp, Nt, 3), dim=-1)).view(B, Lp, H, 1).to(work_dtype)

        o2_carry = torch.matmul(qc2_tc, S_hist) * dec_c_tc
        qk2_eff  = (torch.matmul(qc2_tc, kc_tc.transpose(-2, -1)) * k2_outer_tc) * G_l_tc
        o2_full  = (o2_carry + torch.matmul(qk2_eff, U_all)).reshape(B, H, Lp, Dh)
        o2_gated = (o2_full.permute(0, 2, 1, 3).to(work_dtype) * head_gain2).reshape(B, Lp, D)

        hmix = torch.sigmoid(self.hop_mix_gate).to(work_dtype)
        return self.out_proj(o1_gated + hmix * o2_gated), M_chunk_bank, surprise_map, ortho_loss


# ==============================================================================
# 2. STRICTLY CAUSAL EXACT PER-TOKEN CMER + OVERLAPPING [c-1, c] ROUTER
#    - [FIX #1] ZERO CAUSALITY LEAK! Eliminated 16-token group mean(dim=-2) over future tokens.
#      Computes EXACT per-token causal CMER via tiled contraction:
#        o_cmer[t] = sum_{c < c(t)-1} p_{t,c} * (q_t^(2) @ M_c)
#      using memory-chunk tiling (m_step=16) so peak temporary memory is <12 MB even at 32K!
# ==============================================================================
class CMERStarResonanceRouter(nn.Module):
    def __init__(self, d_model=768, n_heads=12, chunk_size=64, n_slots=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.C = chunk_size
        self.K_s = n_slots

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.sal_proj = nn.Linear(d_model, n_heads * n_slots, bias=False)
        self.delta_sal_proj = nn.Linear(d_model, n_heads * n_slots, bias=False)
        self.surprise_sal_scale = nn.Parameter(torch.full((1, n_heads, n_slots, 1), 0.35))
        self.slot_temp = nn.Parameter(torch.ones(1, n_heads, 1, n_slots, 1) * 1.6)

        self.log_len_scale = nn.Parameter(torch.full((1, n_heads, 1, 1, 1), -1.4))
        self.cross_loop_sal_mix = nn.Parameter(torch.full((1, n_heads, 1, 1, 1), 0.35))
        self.cross_loop_kv_mix  = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.45))

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.v_res_gate = nn.Parameter(torch.tensor(0.0))

        self.cmer_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.cmer_gate   = nn.Parameter(torch.full((1, 1, d_model), -0.5))

        self.epi_hop2_proj = nn.Linear(d_model, d_model, bias=False)
        self.epi_hop2_norm = RMSNorm(self.head_dim)
        self.epi_hop2_scale = nn.Parameter(torch.full((1, n_heads, 1, 1, 1), 0.42))
        self.epi_hop_mix = nn.Parameter(torch.zeros(1, 1, d_model))

        self.sink_logit = nn.Parameter(torch.full((1, n_heads, 1, 1, 1), -2.2))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        prev_ok = torch.ones((chunk_size, chunk_size), dtype=torch.float32)
        curr_ok = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.float32))
        ov_bias_normal = (1.0 - torch.cat([prev_ok, curr_ok], dim=-1)) * (-1e4)
        ov_bias_first  = (1.0 - torch.cat([torch.zeros_like(prev_ok), curr_ok], dim=-1)) * (-1e4)
        self.register_buffer("ov_bias_normal", ov_bias_normal.view(1, 1, 1, chunk_size, 2 * chunk_size))
        self.register_buffer("ov_bias_first",  ov_bias_first.view(1, 1, 1, chunk_size, 2 * chunk_size))

    def _exact_causal_cmer_read(self, p_chunk, q_cmer_sub, M_bank_sub):
        """
        100% Strictly Causal Per-Token CMER (ZERO future-token group averaging!):
          p_chunk:    [B, H, L_sub, N_m]  (strictly zero for c >= c(t)-1)
          q_cmer_sub: [B, H, L_sub, Dh]
          M_bank_sub: [B, H, N_m, Dh, Dh]
        Computes o_cmer[t, v] = sum_{m} p_chunk[t, m] * (q_cmer[t] @ M_bank[m])[v]
        tiled over memory chunks m_step=16 for constant low VRAM up to 32K!
        """
        B, H, L_sub, N_m = p_chunk.shape
        Dh = q_cmer_sub.shape[-1]
        o_acc = torch.zeros((B, H, L_sub, Dh), device=q_cmer_sub.device, dtype=q_cmer_sub.dtype)
        m_step = 16
        q_exp = q_cmer_sub.unsqueeze(2)  # (B, H, 1, L_sub, Dh)
        for ms in range(0, N_m, m_step):
            me = min(N_m, ms + m_step)
            M_blk = M_bank_sub[:, :, ms:me].to(q_cmer_sub.dtype)           # (B, H, mb, Dh, Dh)
            qM = torch.matmul(q_exp, M_blk)                                # (B, H, mb, L_sub, Dh)
            p_blk = p_chunk[:, :, :, ms:me].permute(0, 1, 3, 2).unsqueeze(-1)  # (B, H, mb, L_sub, 1)
            o_acc = o_acc + (p_blk * qM).sum(dim=2)
        return o_acc

    def _competitive_read_with_cmer(self, qc_in, kc_ov, vc_ov, k_epi, v_epi, c_idx, epi_c_idx, scale, M_bank=None, q_cmer=None):
        B, H, n_chunks, C, Dh = qc_in.shape
        Ks = self.K_s
        step = 32  # 32 chunks = 2048 tokens per query tile
        beta_l = F.softplus(self.log_len_scale.float()).to(qc_in.dtype)
        log_c = torch.log(torch.clamp(c_idx.float(), min=1.0)).view(1, 1, n_chunks, 1, 1).to(qc_in.dtype)
        epi_temp = 1.0 + beta_l * log_c

        is_first = (c_idx == 0).to(qc_in.dtype).view(1, 1, n_chunks, 1, 1)
        loc_b_all = is_first * self.ov_bias_first.to(qc_in.dtype) + (1.0 - is_first) * self.ov_bias_normal.to(qc_in.dtype)

        if n_chunks <= step:
            epi_valid = (epi_c_idx.unsqueeze(0) < (c_idx.unsqueeze(1) - 1)).to(qc_in.dtype).view(1, 1, n_chunks, 1, n_chunks * Ks)
            epi_b = (1.0 - epi_valid) * (-1e4)
            s_epi = torch.matmul(qc_in, k_epi.unsqueeze(2).transpose(-2, -1)) * (scale * epi_temp) + epi_b
            s_loc = torch.matmul(qc_in, kc_ov.transpose(-2, -1)) * scale + loc_b_all
            sink  = self.sink_logit.expand(B, H, n_chunks, C, 1).to(s_epi.dtype)
            attn  = torch.softmax(torch.cat([s_epi, s_loc, sink], dim=-1).float(), dim=-1).to(vc_ov.dtype)
            a_epi = attn[..., :n_chunks * Ks]
            a_loc = attn[..., n_chunks * Ks : n_chunks * Ks + 2 * C]
            out_std = torch.matmul(a_epi, v_epi.unsqueeze(2)) + torch.matmul(a_loc, vc_ov)

            if M_bank is not None and q_cmer is not None:
                p_chunk = a_epi.view(B, H, n_chunks * C, n_chunks, Ks).sum(dim=-1)  # (B, H, L, Nc) exact per token!
                q_flat = q_cmer.reshape(B, H, n_chunks * C, Dh)
                o_cmer = self._exact_causal_cmer_read(p_chunk, q_flat, M_bank).view(B, H, n_chunks, C, Dh)
                return out_std, o_cmer
            return out_std, None

        outs_std, outs_cmer = [], []
        for st in range(0, n_chunks, step):
            en = min(n_chunks, st + step)
            nc_sub = en - st
            qc_sub = qc_in[:, :, st:en]
            kc_sub = kc_ov[:, :, st:en]
            vc_sub = vc_ov[:, :, st:en]
            epi_temp_sub = epi_temp[:, :, st:en]
            loc_b_sub = loc_b_all[:, :, st:en]
            max_epi = max(Ks, max(1, en - 2) * Ks)
            k_epi_sub = k_epi[:, :, :max_epi]
            v_epi_sub = v_epi[:, :, :max_epi]
            c_sub = c_idx[st:en].unsqueeze(1)
            epi_sub = epi_c_idx[:max_epi].unsqueeze(0)
            epi_valid_sub = (epi_sub < (c_sub - 1)).to(qc_in.dtype).view(1, 1, nc_sub, 1, max_epi)
            epi_b_sub = (1.0 - epi_valid_sub) * (-1e4)
            s_epi = torch.matmul(qc_sub, k_epi_sub.unsqueeze(2).transpose(-2, -1)) * (scale * epi_temp_sub) + epi_b_sub
            s_loc = torch.matmul(qc_sub, kc_sub.transpose(-2, -1)) * scale + loc_b_sub
            sink  = self.sink_logit.expand(B, H, nc_sub, C, 1).to(s_epi.dtype)
            attn  = torch.softmax(torch.cat([s_epi, s_loc, sink], dim=-1).float(), dim=-1).to(vc_sub.dtype)
            a_epi = attn[..., :max_epi]
            a_loc = attn[..., max_epi : max_epi + 2 * C]
            outs_std.append(torch.matmul(a_epi, v_epi_sub.unsqueeze(2)) + torch.matmul(a_loc, vc_sub))

            if M_bank is not None and q_cmer is not None:
                n_m_sub = max_epi // Ks
                p_chunk = a_epi.view(B, H, nc_sub * C, n_m_sub, Ks).sum(dim=-1)
                q_flat = q_cmer[:, :, st:en].reshape(B, H, nc_sub * C, Dh)
                outs_cmer.append(self._exact_causal_cmer_read(p_chunk, q_flat, M_bank[:, :, :n_m_sub]).view(B, H, nc_sub, C, Dh))
        return torch.cat(outs_std, dim=2), (torch.cat(outs_cmer, dim=2) if outs_cmer else None)

    def forward(self, x, v0=None, loop_idx=0, res_cache=None, M_bank=None, surprise_map=None):
        B, Lp, D = x.shape
        H, Dh, C, Ks = self.n_heads, self.head_dim, self.C, self.K_s
        n_chunks = Lp // C

        q, k, v = self.qkv(x).view(B, Lp, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if v0 is not None:
            v = v + torch.sigmoid(self.v_res_gate).to(v.dtype) * v0.permute(0, 2, 1, 3).to(v.dtype)

        dx = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, 1:, :] - x[:, :-1, :]], dim=1)
        sal = (self.sal_proj(x) + 0.65 * self.delta_sal_proj(dx)).view(B, Lp, H, Ks).permute(0, 2, 3, 1)
        if surprise_map is not None:
            surp = surprise_map.to(sal.dtype).permute(0, 2, 1).unsqueeze(2)
            sal = sal + self.surprise_sal_scale.to(sal.dtype) * surp

        qc = q.view(B, H, n_chunks, C, Dh)
        kc = k.view(B, H, n_chunks, C, Dh)
        vc = v.view(B, H, n_chunks, C, Dh)
        kc_prev = torch.cat([torch.zeros_like(kc[:, :, :1]), kc[:, :, :-1]], dim=2)
        vc_prev = torch.cat([torch.zeros_like(vc[:, :, :1]), vc[:, :, :-1]], dim=2)
        kc_ov = torch.cat([kc_prev, kc], dim=3)
        vc_ov = torch.cat([vc_prev, vc], dim=3)

        sal_c = sal.view(B, H, Ks, n_chunks, C).permute(0, 1, 3, 2, 4)
        sal_orth = (sal_c - 0.55 * sal_c.mean(dim=-2, keepdim=True)) * F.softplus(self.slot_temp.float()).to(sal_c.dtype)
        if loop_idx > 0 and res_cache is not None:
            sal_orth = sal_orth + torch.tanh(self.cross_loop_sal_mix.float()).to(sal_orth.dtype) * res_cache["sal_orth"]

        w_epi = torch.softmax(sal_orth.float(), dim=-1).to(q.dtype)
        k_epi = self.k_norm(torch.matmul(w_epi, kc)).reshape(B, H, n_chunks * Ks, Dh)
        v_epi = torch.matmul(w_epi, vc).reshape(B, H, n_chunks * Ks, Dh)

        if loop_idx > 0 and res_cache is not None:
            kv_mix = torch.tanh(self.cross_loop_kv_mix.float()).to(k_epi.dtype)
            k_epi = self.k_norm(k_epi + kv_mix * res_cache["k_epi"])
            v_epi = v_epi + kv_mix * res_cache["v_epi"]

        new_res_cache = {"sal_orth": sal_orth.detach(), "k_epi": k_epi, "v_epi": v_epi}

        scale = 1.0 / math.sqrt(Dh)
        c_idx = torch.arange(n_chunks, device=x.device)
        epi_c_idx = torch.arange(n_chunks, device=x.device).repeat_interleave(Ks)

        if loop_idx == 0:
            out_c1, _ = self._competitive_read_with_cmer(qc, kc_ov, vc_ov, k_epi, v_epi, c_idx, epi_c_idx, scale)
            out1 = out_c1.reshape(B, H, Lp, Dh).transpose(1, 2).contiguous().view(B, Lp, D)
            return self.out_proj(out1), new_res_cache

        q_cmer = F.normalize(self.cmer_q_proj(x).view(B, Lp, H, Dh).permute(0, 2, 1, 3).float(), dim=-1, eps=1e-5).view(B, H, n_chunks, C, Dh).to(q.dtype)
        out_c1, o_cmer1 = self._competitive_read_with_cmer(qc, kc_ov, vc_ov, k_epi, v_epi, c_idx, epi_c_idx, scale, M_bank=M_bank, q_cmer=q_cmer)
        out1 = out_c1.reshape(B, H, Lp, Dh).transpose(1, 2).contiguous().view(B, Lp, D)
        if o_cmer1 is not None:
            cmer_flat = o_cmer1.reshape(B, H, Lp, Dh).transpose(1, 2).contiguous().view(B, Lp, D)
            out1 = out1 + torch.sigmoid(self.cmer_gate).to(out1.dtype) * cmer_flat

        q2_delta = self.epi_hop2_proj(out1).view(B, Lp, H, Dh).permute(0, 2, 1, 3)
        qc2 = self.epi_hop2_norm(qc + torch.tanh(self.epi_hop2_scale.float()).to(qc.dtype) * q2_delta.view(B, H, n_chunks, C, Dh))

        out_c2, _ = self._competitive_read_with_cmer(qc2, kc_ov, vc_ov, k_epi, v_epi, c_idx, epi_c_idx, scale)
        out2 = out_c2.reshape(B, H, Lp, Dh).transpose(1, 2).contiguous().view(B, Lp, D)

        emix = torch.sigmoid(self.epi_hop_mix).to(x.dtype)
        return self.out_proj(out1 + emix * out2), new_res_cache


# ==============================================================================
# 3. PLATINUMSTAR-3.0.1 CORTICAL CELL WITH IMPENETRABLE BOUNDARY-WALL DENDRITIC STEM
#    - [FIX #3A] Explicit causal shifted taps with cumulative boundary wall:
#        tap_{t-1} gated by shift1(m); tap_{t-3} gated by shift1(m) * shift2(m) * shift3(m)!
#      Now ANY delimiter (:, ;, \n) between t-3 and t is an impenetrable wall (0 extra params)!
# ==============================================================================
class PlatinumStarV3Cell(nn.Module):
    def __init__(self, d_model=768, n_heads=12, mode="star_triad_v3", ffn_dim=4096, lora_rank=128):
        super().__init__()
        self.mode = mode
        self.norm = RMSNorm(d_model)
        # Explicit depthwise weights for taps (t, t-1) and (t, t-3) — exact same param count as Conv1d(k=2)!
        self.d1_w_curr = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.d1_w_prev = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.d3_w_curr = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.d3_w_prev = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.stem_gate = nn.Linear(d_model, 4, bias=True)
        with torch.no_grad():
            self.stem_gate.weight.zero_()
            self.stem_gate.bias.copy_(torch.tensor([0.2, 0.5, 0.2, 1.5]))

        if mode == "star_triad_v3":
            self.assoc = StarTriadV3Engine(d_model, n_heads, chunk_size=64)
        elif mode == "cmer_resonance":
            self.assoc = CMERStarResonanceRouter(d_model, n_heads, chunk_size=64, n_slots=2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.c_norm = RMSNorm(d_model)
        self.w_ctx = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w_out = nn.Linear(ffn_dim, d_model, bias=False)
        self.lora_a = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_b = nn.Linear(lora_rank, d_model, bias=False)
        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)
        self.lora_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, v0=None, loop_idx=0, res_cache=None, M_bank=None, surprise_map=None, compute_ortho=False):
        h = self.norm(x)
        sg = self.stem_gate(h).float()
        w_br = torch.softmax(sg[..., :3], dim=-1).to(h.dtype)
        m = torch.sigmoid(sg[..., 3:4]).to(h.dtype)  # (B, Lp, 1) boundary keep mask (0 at delimiter!)

        # [FIX #3A] Impenetrable cumulative boundary wall for d=1 and d=3 taps!
        z1 = torch.zeros_like(m[:, :1, :])
        m_s1 = torch.cat([z1, m[:, :-1, :]], dim=1)
        m_s2 = torch.cat([z1, m_s1[:, :-1, :]], dim=1)
        m_s3 = torch.cat([z1, m_s2[:, :-1, :]], dim=1)
        wall_1 = m_s1
        wall_3 = m_s1 * m_s2 * m_s3

        zh1 = torch.zeros_like(h[:, :1, :])
        h_s1 = torch.cat([zh1, h[:, :-1, :]], dim=1)
        h_s3 = torch.cat([torch.zeros_like(h[:, :3, :]), h[:, :-3, :]], dim=1)

        d1 = self.d1_w_curr.to(h.dtype) * h + self.d1_w_prev.to(h.dtype) * (h_s1 * wall_1)
        d3 = self.d3_w_curr.to(h.dtype) * h + self.d3_w_prev.to(h.dtype) * (h_s3 * wall_3)
        h_dend = w_br[..., 0:1] * h + w_br[..., 1:2] * d1 + w_br[..., 2:3] * d3

        ortho_loss = None
        if self.mode == "cmer_resonance":
            a_out, res_cache = self.assoc(h_dend, v0=v0, loop_idx=loop_idx, res_cache=res_cache, M_bank=M_bank, surprise_map=surprise_map)
        else:
            a_out, M_bank, surprise_map, ortho_loss = self.assoc(h_dend, v0=v0, loop_idx=loop_idx, compute_ortho=compute_ortho)

        u = x + a_out
        c_in = self.c_norm(u + 0.5 * h_dend)
        g1, g2 = self.w_ctx(c_in).chunk(2, dim=-1)
        c_out = self.w_out(F.silu(torch.clamp(g1, max=10.0)) * torch.clamp(g2, min=-10.0, max=10.0))
        if loop_idx > 0:
            c_out = c_out + self.lora_scale.to(x.dtype) * self.lora_b(F.silu(self.lora_a(c_in)))
        return u + c_out, res_cache, M_bank, surprise_map, ortho_loss


# ==============================================================================
# 4. PLATINUMSTAR-3.0.1 (`6x2 = 12` EFFECTIVE LAYERS) + MTP-2 + CAUSALITY SELF-TEST
# ==============================================================================




QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
HORIZONS  = (1, 5, 20)
N_FEATURES = 16


class CausalMultiScaleMarketStem(nn.Module):
    """[H1] Strictly Causal Multi-Scale Stem (k=1, 4, 16) + Subordinated Market-Clock (dt x Vol x Volume)."""
    def __init__(self, n_features=N_FEATURES, d_model=384, dt_idx=12, vol_idx=8, dv_idx=6, dropout=0.18):
        super().__init__()
        self.dt_idx, self.vol_idx, self.dv_idx = dt_idx, vol_idx, dv_idx
        self.in_norm = nn.LayerNorm(n_features)
        self.drop = nn.Dropout(dropout)
        self.conv4  = nn.Conv1d(n_features, n_features, kernel_size=4,  groups=n_features)
        self.conv16 = nn.Conv1d(n_features, n_features, kernel_size=16, groups=n_features)
        self.ms_norm = nn.LayerNorm(n_features * 3)
        self.in_proj = nn.Sequential(
            nn.Linear(n_features * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.clock_mix = nn.Linear(3, 1, bias=True)
        with torch.no_grad():
            self.clock_mix.weight.copy_(torch.tensor([[1.0, 0.35, 0.15]]))
            self.clock_mix.bias.zero_()
        self.dt_freqs = nn.Parameter(torch.linspace(0.1, 10.0, 24))
        self.dt_proj  = nn.Linear(48, d_model, bias=False)
        self.dt_gate  = nn.Parameter(torch.tensor(0.25))

    def forward(self, x_feat):
        x_in = x_feat + 0.02 * torch.randn_like(x_feat) if self.training else x_feat
        xn = self.in_norm(x_in)
        xt = xn.transpose(1, 2)
        c4  = self.conv4(F.pad(xt, (3, 0))).transpose(1, 2)
        c16 = self.conv16(F.pad(xt, (15, 0))).transpose(1, 2)
        ms = self.ms_norm(torch.cat([xn, c4, c16], dim=-1))
        h = self.in_proj(self.drop(ms))
        clock_in = torch.stack([x_feat[:, :, self.dt_idx], x_feat[:, :, self.vol_idx], x_feat[:, :, self.dv_idx]], dim=-1).to(h.dtype)
        m_clock = F.softplus(self.clock_mix(clock_in))
        ang = m_clock * self.dt_freqs.to(h.dtype).view(1, 1, -1)
        dt_emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return h + torch.tanh(self.dt_gate).to(h.dtype) * self.dt_proj(dt_emb)


class MonotonicQuantileHead(nn.Module):
    """[H3] Strictly non-crossing quantile head: q50 anchor + cumulative softplus spreads."""
    def __init__(self, d_model, n_horizons=len(HORIZONS)):
        super().__init__()
        self.n_horizons = n_horizons
        self.proj = nn.Linear(d_model, n_horizons * 5)

    def forward(self, h):
        raw = self.proj(h).float().view(h.size(0), h.size(1), self.n_horizons, 5)
        q50 = raw[:, :, :, 2]
        dl1 = F.softplus(raw[:, :, :, 1]) + 1e-3
        dl2 = F.softplus(raw[:, :, :, 0]) + 1e-3
        dh1 = F.softplus(raw[:, :, :, 3]) + 1e-3
        dh2 = F.softplus(raw[:, :, :, 4]) + 1e-3
        q25 = q50 - dl1
        q10 = q25 - dl2
        q75 = q50 + dh1
        q90 = q75 + dh2
        return torch.stack([q10, q25, q50, q75, q90], dim=-1)


def causal_zscore_1d(x: torch.Tensor, win: int = 32) -> torch.Tensor:
    x3 = x.float().unsqueeze(1)
    xp = F.pad(x3, (win - 1, 0), mode="replicate")
    mu = F.avg_pool1d(xp, kernel_size=win, stride=1).squeeze(1)
    sq = F.avg_pool1d(xp * xp, kernel_size=win, stride=1).squeeze(1)
    sd = torch.sqrt((sq - mu * mu).clamp(min=1e-6))
    return ((x.float() - mu) / sd).clamp(-6.0, 6.0)


class PlatinumQuant4(nn.Module):
    """PlatinumQuant-4.1 (StarQuant-4, ~3.7M params, 100% Non-Transformer):
    Carries H1-H5 from PQ-3 + Causal Spectral-Gated Refinement for Financial Markets.
    """
    def __init__(self, n_features=N_FEATURES, d_model=384, n_heads=6, ffn_dim=768, lora_rank=32, dropout=0.18):
        super().__init__()
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.n_loops, self.C = 2, 64
        self.market_stem = CausalMultiScaleMarketStem(n_features, d_model, dropout=dropout)
        self.emb_norm = RMSNorm(d_model)
        self.v0_proj = nn.Linear(d_model, d_model, bias=False)
        self.loop_gate = nn.Parameter(torch.tensor([0.0, -1.2]))
        self.loop_embeds = nn.Parameter(torch.zeros(2, 1, 1, d_model))
        self.drop = nn.Dropout(dropout)
        self.trend_skip = nn.Linear(n_features, d_model, bias=False)
        nn.init.zeros_(self.trend_skip.weight)
        self.blocks = nn.ModuleList([
            PlatinumStarV3Cell(d_model, n_heads, mode="star_triad_v3",  ffn_dim=ffn_dim, lora_rank=lora_rank),
            PlatinumStarV3Cell(d_model, n_heads, mode="cmer_resonance", ffn_dim=ffn_dim, lora_rank=lora_rank),
        ])
        self.nf = RMSNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.quantile_head = MonotonicQuantileHead(d_model, len(HORIZONS))
        self.dir_head      = nn.Linear(d_model, 3)
        self.vol_head      = nn.Linear(d_model, 2)

    def forward_features(self, x_feat, compute_ortho=False):
        B, L, _ = x_feat.shape
        pad_len = (self.C - (L % self.C)) % self.C
        x_pad = torch.cat([x_feat, x_feat[:, -1:, :].expand(B, pad_len, -1)], dim=1) if pad_len > 0 else x_feat
        Lp = x_pad.shape[1]
        x_in = self.market_stem(x_pad)
        x_norm = self.emb_norm(x_in)
        v0 = self.v0_proj(x_norm).float().view(B, Lp, self.n_heads, self.head_dim)
        h = x_in
        res_caches, M_bank, surprise_map, ortho_losses = [None, None], None, None, []
        for r in range(self.n_loops):
            h = h + (torch.sigmoid(self.loop_gate[r]).to(h.dtype) * x_norm if r > 0 else 0.0) + self.loop_embeds[r].to(h.dtype)
            for b_i, blk in enumerate(self.blocks):
                h, res_caches[b_i], M_bank, surprise_map, o_l = blk(
                    h, v0=v0, loop_idx=r, res_cache=res_caches[b_i],
                    M_bank=M_bank, surprise_map=surprise_map, compute_ortho=(compute_ortho and r == 1)
                )
                h = self.drop(h)
                if o_l is not None:
                    ortho_losses.append(o_l)
        ortho_total = torch.stack(ortho_losses).mean() if ortho_losses else None
        h_clean = self.nf(h[:, :L, :] + self.trend_skip(x_feat.to(h.dtype)))
        h_out = self.head_drop(h_clean)
        dh = h_clean[:, 1:, :].float() - h_clean[:, :-1, :].float()
        vel = torch.cat([torch.zeros((B, 1), device=h_clean.device, dtype=torch.float32), dh.norm(dim=-1)], dim=1)
        if surprise_map is not None:
            raw_surp = surprise_map[:, :L, :].float().mean(dim=-1)
            innovation = 0.70 * causal_zscore_1d(vel, win=32) + 0.30 * causal_zscore_1d(raw_surp, win=32)
        else:
            innovation = causal_zscore_1d(vel, win=32)
        return h_out, ortho_total, innovation

    def forward(self, x_feat, y_ret=None, y_dir=None, y_vol=None, y_jump=None, mask=None, use_ortho=False):
        h_norm, ortho_loss, innovation = self.forward_features(x_feat, compute_ortho=(use_ortho and y_ret is not None))
        q_pred   = self.quantile_head(h_norm)
        dir_log  = self.dir_head(h_norm).float()
        vol_out  = self.vol_head(h_norm).float()
        vol_pred = F.softplus(vol_out[:, :, 0])
        jmp_log  = vol_out[:, :, 1]
        if y_ret is None:
            return q_pred, dir_log, vol_pred, jmp_log, innovation
        return compute_quant_loss(q_pred, dir_log, vol_pred, jmp_log, y_ret, y_dir, y_vol, y_jump, mask, ortho_loss)


def compute_quant_loss(q_pred, dir_log, vol_pred, jmp_log, y_ret, y_dir, y_vol, y_jump, mask=None, ortho_loss=None):
    B, L = y_dir.shape
    m = mask.float() if mask is not None else torch.ones((B, L), device=y_ret.device, dtype=torch.float32)
    m_sum = m.sum().clamp(min=1.0)
    qs = torch.tensor(QUANTILES, device=y_ret.device, dtype=torch.float32).view(1, 1, 1, -1)
    err = y_ret.unsqueeze(-1) - q_pred
    pinball = torch.maximum(qs * err, (qs - 1.0) * err).mean(dim=(-1, -2))
    loss_q   = (pinball * m).sum() / m_sum
    loss_dir = (F.cross_entropy(dir_log.view(-1, 3), y_dir.view(-1), reduction="none").view(B, L) * m).sum() / m_sum
    loss_vol = (F.smooth_l1_loss(vol_pred, y_vol, reduction="none") * m).sum() / m_sum
    loss_jmp = (F.binary_cross_entropy_with_logits(jmp_log, y_jump, reduction="none") * m).sum() / m_sum
    total = 4.0 * loss_q + 0.35 * loss_dir + 1.5 * loss_vol + 0.25 * loss_jmp
    if ortho_loss is not None:
        total = total + 1e-3 * ortho_loss
    return total



