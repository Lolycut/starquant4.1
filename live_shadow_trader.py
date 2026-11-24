"""PlatinumQuant-4.1 Live Hourly Shadow Paper-Trader (GitHub Actions / CPU / GPU).
- Loads 5-seed FP16 ensemble checkpoint (pq41_ensemble_5seeds_fp16.pt)
- Fetches latest strictly CLOSED 1h candles via yfinance (FX excluded per stress-test)
- Applies H5 Uncertainty-Adjusted Conviction Gate + Hawkes Jump Crash Brake + 5 bps fee
- Logs every hourly decision & realized PnL to data/paper_trades_log.csv and data/portfolio_state.json
- Optionally sends concise hourly digest to Telegram (TG_BOT_TOKEN & TG_CHAT_ID env vars)
"""
import json
import math
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import yfinance as yf

from pq41_model import PlatinumQuant4, QUANTILES, HORIZONS

# Trading Universe (FX excluded per Institutional Stress-Test: negative Rank-IC -0.0368)
TRADING_UNIVERSE = {
    "US_Indices":  ["SPY", "QQQ", "IWM"],
    "CommodBonds": ["GLD", "SLV", "TLT"],
    "MegaCapTech": ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META", "AMD", "GOOGL"],
    "ValueEnergy": ["JPM", "XOM"],
    "Crypto":      ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"],
}
ALL_TICKERS = [t for grp in TRADING_UNIVERSE.values() for t in grp]
TICKER_CLASS = {t: cls for cls, grp in TRADING_UNIVERSE.items() for t in grp}

CKPT_PATH = os.environ.get("PQ41_CKPT", "pq41_ensemble_5seeds_fp16.pt")
STATE_PATH = "data/portfolio_state.json"
LOG_CSV = "data/paper_trades_log.csv"
FEE_BPS = 0.0005  # 5 bps realistic taker fee
DAILY_DD_STOP = -0.02  # -2% daily drawdown circuit breaker
HAWKES_BRAKE_THR = 0.70  # Halve position size if jump probability > 70%


def build_live_causal_features(df: pd.DataFrame):
    """Build exact 16 causal features on strictly CLOSED 1h candles."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 120:
        return None, None, None
    # Drop the latest bar if it might still be forming (strictly closed candles only!)
    df = df.iloc[:-1]
    o, h, l, c = [df[k].astype(np.float64).values for k in ("Open", "High", "Low", "Close")]
    v = np.maximum(df["Volume"].astype(np.float64).values if "Volume" in df.columns else np.ones_like(c), 1.0)
    eps = 1e-8
    log_c = np.log(np.maximum(c, eps))
    ret1 = np.zeros_like(log_c)
    ret1[1:] = np.clip(log_c[1:] - log_c[:-1], -0.25, 0.25)

    s_ret = pd.Series(ret1)
    vol10 = s_ret.rolling(10, min_periods=1).std().fillna(0.01).values + 1e-4
    vol60 = s_ret.rolling(60, min_periods=1).std().fillna(0.01).values + 1e-4
    ret_z  = np.clip(ret1 / vol60, -6.0, 6.0)
    hl_rng = np.clip((h - l) / (np.maximum(c, eps) * vol60), 0.0, 10.0)
    co_bdy = np.clip((c - o) / (np.maximum(c, eps) * vol60), -6.0, 6.0)
    up_wk  = np.clip((h - np.maximum(o, c)) / (np.maximum(c, eps) * vol60), 0.0, 8.0)
    lo_wk  = np.clip((np.minimum(o, c) - l) / (np.maximum(c, eps) * vol60), 0.0, 8.0)

    lv_s = pd.Series(np.log1p(v))
    lv_z = np.clip(((lv_s - lv_s.rolling(60, min_periods=1).mean()) / (lv_s.rolling(60, min_periods=1).std() + 1e-4)).fillna(0.0).values, -5.0, 5.0)
    dv_s = pd.Series(np.log1p(v * np.maximum(c, eps)))
    dv_z = np.clip(((dv_s - dv_s.rolling(60, min_periods=1).mean()) / (dv_s.rolling(60, min_periods=1).std() + 1e-4)).fillna(0.0).values, -5.0, 5.0)

    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    vwap_dev = np.clip((c - ema20) / (np.maximum(c, eps) * vol60), -6.0, 6.0)
    macd_n   = np.clip((ema20 - ema50) / (np.maximum(c, eps) * vol60), -6.0, 6.0)

    gains  = pd.Series(np.maximum(ret1, 0.0)).ewm(span=14, adjust=False).mean().values
    losses = pd.Series(np.maximum(-ret1, 0.0)).ewm(span=14, adjust=False).mean().values + 1e-8
    rsi_n  = (gains / (gains + losses) - 0.5) * 2.0

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        ts_h = idx.view("int64") / 3.6e12
        dt_h = np.ones_like(ts_h)
        dt_h[1:] = np.maximum(ts_h[1:] - ts_h[:-1], 1e-3)
        log_dt = np.clip(np.log1p(dt_h), 0.0, 8.0)
        hr, dow = idx.hour.values, idx.dayofweek.values
    else:
        log_dt = np.ones_like(c) * math.log(2.0)
        hr, dow = np.arange(len(c)) % 24, (np.arange(len(c)) // 24) % 7

    feats = np.stack([
        ret_z, hl_rng, co_bdy, up_wk, lo_wk, lv_z, dv_z, vwap_dev,
        np.clip(vol10 / vol60, 0.0, 5.0), np.clip(np.log(vol60 + 1e-5) + 5.0, -5.0, 5.0),
        rsi_n, macd_n, log_dt,
        np.sin(2*math.pi*hr/24.0), np.cos(2*math.pi*hr/24.0), np.sin(2*math.pi*dow/7.0),
    ], axis=-1).astype(np.float32)

    # Keep last 512 closed bars for inference
    feats = feats[-512:]
    last_close = float(c[-1])
    last_ts = str(idx[-1])
    return feats, last_close, last_ts


def load_ensemble(ckpt_path: str, device: str):
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    models = []
    for sd in bundle["state_dicts"]:
        m = PlatinumQuant4().to(device).eval()
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        models.append(m)
    return models, bundle.get("seeds", [])


def send_telegram(msg: str):
    token = os.environ.get("TG_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"[Telegram warning] {e}")


def main():
    os.makedirs("data", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== PlatinumQuant-4.1 Live Shadow Run | {now_utc} | Device: {device} ===")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Missing ensemble checkpoint '{CKPT_PATH}'. "
            "Run pq41_train_export_5seed_ensemble.ipynb on Kaggle first and place the .pt file in the repo root."
        )

    models, seeds = load_ensemble(CKPT_PATH, device)
    print(f"Loaded {len(models)}-seed ensemble (seeds={seeds})")

    if os.path.exists(STATE_PATH):
        state = json.load(open(STATE_PATH, "r", encoding="utf-8"))
    else:
        state = {"equity": 1.0, "peak_equity": 1.0, "positions": {}, "last_prices": {}, "last_ts": {}, "n_runs": 0}

    rows = []
    step_pnls = []
    active_signals = []

    for tk in ALL_TICKERS:
        try:
            df = yf.download(tk, period="60d", interval="1h", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            feats, price, bar_ts = build_live_causal_features(df)
            if feats is None:
                continue
        except Exception as e:
            print(f"  [Skip {tk}] download error: {e}")
            continue

        # Skip if this ticker has not closed a new bar since last run (e.g. equities on weekends/nights)
        prev_ts = state["last_ts"].get(tk)
        prev_pos = float(state["positions"].get(tk, 0.0))
        prev_price = state["last_prices"].get(tk)

        bar_pnl = 0.0
        if prev_ts is not None and bar_ts != prev_ts and prev_price is not None and prev_price > 0:
            raw_ret = math.log(price / prev_price)
            bar_pnl = prev_pos * raw_ret

        # Run 5-seed ensemble on CPU/GPU
        x_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        q_acc, d_acc, j_acc = [], [], []
        with torch.inference_mode():
            for m in models:
                qp, dl, _, jl, _ = m(x_t)
                q_acc.append(qp[0, -1, 1].cpu().numpy())          # +5 bar quantiles [q10..q90]
                d_acc.append(F.softmax(dl[0, -1], dim=-1).cpu().numpy())
                j_acc.append(float(torch.sigmoid(jl[0, -1]).item()))

        q_mean = np.mean(q_acc, axis=0)
        d_mean = np.mean(d_acc, axis=0)
        p_jump = float(np.mean(j_acc))
        q10, q50, q90 = float(q_mean[0]), float(q_mean[2]), float(q_mean[4])
        ci_w = max(q90 - q10, 0.25)

        # H5 Uncertainty-Adjusted Conviction Gate (slightly looser entry 0.22 for Crypto)
        conv = 0.55 * float(d_mean[2] - d_mean[0]) + 0.45 * math.tanh(q50 / (0.5 * ci_w))
        ent_thr = 0.16 if TICKER_CLASS[tk] == "Crypto" else 0.18
        ext_thr = 0.04

        new_pos = prev_pos
        if bar_ts != prev_ts:
            if abs(prev_pos) < 1e-6:
                if conv > ent_thr: new_pos = 1.0
                elif conv < -ent_thr: new_pos = -1.0
            elif prev_pos > 0 and conv < ext_thr:
                new_pos = -1.0 if conv < -ent_thr else 0.0
            elif prev_pos < 0 and conv > -ext_thr:
                new_pos = 1.0 if conv > ent_thr else 0.0

            # Hawkes Jump Crash Brake: halve exposure during extreme shock risk
            if p_jump > HAWKES_BRAKE_THR and abs(new_pos) > 0.5:
                new_pos *= 0.5

        dpos = abs(new_pos - prev_pos)
        net_bar_pnl = bar_pnl - dpos * FEE_BPS
        if bar_ts != prev_ts:
            step_pnls.append(net_bar_pnl)

        state["positions"][tk] = round(new_pos, 4)
        state["last_prices"][tk] = price
        state["last_ts"][tk] = bar_ts

        if abs(new_pos) > 0:
            active_signals.append(f"{tk}: {'LONG' if new_pos > 0 else 'SHORT'}({abs(new_pos):.1f}) conv={conv:+.2f} jmp={p_jump:.2f}")

        rows.append({
            "run_utc": now_utc, "ticker": tk, "asset_class": TICKER_CLASS[tk], "bar_ts": bar_ts,
            "price": round(price, 4), "q10": round(q10, 4), "q50": round(q50, 4), "q90": round(q90, 4),
            "p_jump": round(p_jump, 4), "conviction": round(conv, 4),
            "prev_pos": prev_pos, "new_pos": new_pos, "net_pnl_bps": round(net_bar_pnl * 1e4, 2),
        })

    if step_pnls:
        port_ret = float(np.mean(step_pnls))
        state["equity"] = float(state["equity"] * math.exp(port_ret))
        state["peak_equity"] = max(state["peak_equity"], state["equity"])
    else:
        port_ret = 0.0

    state["n_runs"] = int(state.get("n_runs", 0)) + 1
    dd_pct = (state["equity"] / max(state["peak_equity"], 1e-8) - 1.0) * 100.0
    tot_ret_pct = (state["equity"] - 1.0) * 100.0

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    df_new = pd.DataFrame(rows)
    if os.path.exists(LOG_CSV):
        df_new.to_csv(LOG_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_CSV, index=False)

    summary = (
        f"*PlatinumQuant-4.1 Live Paper-Trader* ({now_utc})\n"
        f"Equity: `{state['equity']:.4f}` (`{tot_ret_pct:+.2f}%` | DD: `{dd_pct:.2f}%`)\n"
        f"Last Step: `{port_ret*1e4:+.1f} bps` | Active: `{len(active_signals)}/{len(ALL_TICKERS)}`\n"
        + ("\n".join(active_signals[:8]) if active_signals else "All positions FLAT (cash).")
    )
    print("\n" + summary)
    send_telegram(summary)


if __name__ == "__main__":
    main()
