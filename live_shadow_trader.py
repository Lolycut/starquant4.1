"""PlatinumQuant-4.1 Live Hourly Shadow Paper-Trader (v2 — Hardened Pipeline).
Fixes over v1:
- Strict monotonic bar timestamp guard (eliminates 73% overnight duplicates & yfinance 00:01 UTC rollbacks)
- Time-aware closed-candle filter (fixes 18.5h NYSE 15:30 close-bar lag; tuned for XX:01 UTC cron)
- Local monotonic OHLCV cache per ticker (eliminates Yahoo query1/query2 cache flapping)
- Unit-safe dt_h calculation (fixes deprecated idx.view('int64') / 3.6e12 bug in pandas 2.x/3.x)
- Zero-cold-start causal Rolling Z-Score conviction gate using all 512 causal model outputs
  (removes +0.083/-0.039 bias and unlocks Crypto & symmetric longs/shorts; raw mode via env var)
- Fixed per-asset capital weight (1/N) portfolio accounting + active DAILY_DD_STOP circuit breaker
- Safe automatic schema migration for existing data/paper_trades_log.csv
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import yfinance as yf

from pq41_model import PlatinumQuant4

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
ASSET_WEIGHT = 1.0 / len(ALL_TICKERS)  # Fixed capital slice per asset (consistent day & night)

# Paths & Config (override via env vars without editing code)
CKPT_PATH = os.environ.get("PQ41_CKPT", "pq41_ensemble_5seeds_fp16.pt")
STATE_PATH = os.environ.get("PQ41_STATE", "data/portfolio_state.json")
LOG_CSV = os.environ.get("PQ41_LOG_CSV", "data/paper_trades_log.csv")
CACHE_DIR = os.environ.get("PQ41_CACHE_DIR", "data/ohlcv_cache")

FEE_RATE = float(os.environ.get("PQ41_FEE_RATE", "0.0005"))          # 5 bps taker fee
DAILY_DD_STOP = float(os.environ.get("PQ41_DAILY_DD_STOP", "-0.02")) # -2% UTC-day stop
HAWKES_BRAKE_THR = float(os.environ.get("PQ41_HAWKES_THR", "0.70"))  # Halve size if p_jump > 70%
CLOSE_BUFFER_SEC = int(os.environ.get("PQ41_CLOSE_BUF_SEC", "30"))   # Fits XX:01 UTC cron

# Signal Gate Config: "zscore" (recommended) or "raw" (legacy static thresholds)
GATE_MODE = os.environ.get("PQ41_GATE_MODE", "zscore").strip().lower()
Z_WARMUP = 64                                                         # Skip early window warmup bars
Z_WIN = int(os.environ.get("PQ41_Z_WIN", "240"))                      # Trailing causal bars for z-score
Z_ENTRY = float(os.environ.get("PQ41_Z_ENTRY", "1.50"))               # Enter at |z| > 1.50
Z_EXIT = float(os.environ.get("PQ41_Z_EXIT", "0.40"))                 # Hysteresis exit at |z| < 0.40
MIN_CONV_STD = float(os.environ.get("PQ41_MIN_CONV_STD", "0.015"))    # Floor against flat-regime noise

# Fallback thresholds if GATE_MODE == "raw"
RAW_ENTRY_CRYPTO = float(os.environ.get("PQ41_RAW_ENT_CRYPTO", "0.06"))
RAW_ENTRY_EQUITY = float(os.environ.get("PQ41_RAW_ENT_EQUITY", "0.13"))
RAW_EXIT = float(os.environ.get("PQ41_RAW_EXIT", "0.03"))

TG_NOTIFY_ONLY_ON_NEW_BARS = os.environ.get("TG_NOTIFY_ONLY_ON_NEW_BARS", "0") == "1"

LOG_COLUMNS = [
    "run_utc", "ticker", "asset_class", "bar_ts", "price",
    "q10", "q50", "q90", "p_jump", "conviction", "conv_z",
    "prev_pos", "new_pos", "net_pnl_bps", "port_equity",
]


def parse_utc_ts(ts_val) -> pd.Timestamp | None:
    if ts_val is None or (isinstance(ts_val, float) and math.isnan(ts_val)):
        return None
    ts = pd.Timestamp(ts_val)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def get_exchange_tz(asset_class: str) -> str:
    return "UTC" if asset_class == "Crypto" else "America/New_York"


def drop_unclosed_bars(
    df: pd.DataFrame,
    asset_class: str,
    now_utc: pd.Timestamp,
    buffer_sec: int = CLOSE_BUFFER_SEC,
) -> pd.DataFrame:
    """Keep only strictly CLOSED candles based on actual exchange close time."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    buf = pd.Timedelta(seconds=buffer_sec)

    while len(df) > 0:
        last_ts = df.index[-1]
        last_ts_utc = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
        if asset_class != "Crypto":
            ny_ts = last_ts_utc.tz_convert("America/New_York")
            # NYSE final hourly bar (15:30-16:00 local) is 30 minutes long
            dur = pd.Timedelta(minutes=30) if (ny_ts.hour == 15 and ny_ts.minute == 30) else pd.Timedelta(hours=1)
        else:
            dur = pd.Timedelta(hours=1)

        if now_utc < last_ts_utc + dur + buf:
            df = df.iloc[:-1]
        else:
            break
    return df


def fetch_and_merge_ohlcv(ticker: str, asset_class: str, now_utc: pd.Timestamp) -> pd.DataFrame | None:
    """Download 1h OHLCV via yfinance with retries and merge into immutable local closed-bar cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = ticker.replace("/", "_").replace("=", "_")
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}.csv")
    exch_tz = get_exchange_tz(asset_class)

    cached = None
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0)
            cached.index = pd.to_datetime(cached.index, utc=True)
            cached = cached[~cached.index.duplicated(keep="first")].sort_index()
        except Exception as e:
            print(f"  [Cache warn {ticker}] resetting corrupted cache: {e}")
            cached = None

    # Fetch shorter lookback once cache is warm; 60d on cold start
    period = "7d" if (cached is not None and len(cached) >= 600) else "60d"
    fresh = None
    for attempt in range(3):
        try:
            dl = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
            if dl is not None and not dl.empty:
                if isinstance(dl.columns, pd.MultiIndex):
                    dl.columns = dl.columns.get_level_values(0)
                cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in dl.columns]
                dl = dl[cols].dropna(subset=["Open", "High", "Low", "Close"]).copy()
                dl.index = pd.to_datetime(dl.index, utc=True)
                dl = drop_unclosed_bars(dl, asset_class, now_utc)
                if not dl.empty:
                    fresh = dl
                    break
        except Exception as e:
            if attempt == 2:
                print(f"  [Download error {ticker}] {e}")
            time.sleep(1.5 * (attempt + 1))

    if cached is not None and fresh is not None:
        # Keep previously finalized bars immutable; append ONLY strictly newer closed bars
        # (prevents Yahoo query1/query2 historical revisions & 00:01 UTC rollbacks)
        last_cached_ts = cached.index[-1]
        newer = fresh[fresh.index > last_cached_ts]
        merged = pd.concat([cached, newer]) if not newer.empty else cached
    elif fresh is not None:
        merged = fresh
    elif cached is not None:
        merged = cached
    else:
        return None

    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    # Keep a comfortable rolling buffer (~1200 bars) in local cache
    merged = merged.iloc[-1200:]
    merged.to_csv(cache_path)

    # Restore exact exchange timezone so hour/dayofweek features match training 1:1
    out = merged.copy()
    out.index = out.index.tz_convert(exch_tz)
    return out


def build_live_causal_features(df: pd.DataFrame):
    """Build exact 16 causal features on already-validated strictly CLOSED 1h candles."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 120:
        return None, None, None, None

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
    lv_z = np.clip(
        ((lv_s - lv_s.rolling(60, min_periods=1).mean()) / (lv_s.rolling(60, min_periods=1).std() + 1e-4))
        .fillna(0.0).values,
        -5.0, 5.0,
    )
    dv_s = pd.Series(np.log1p(v * np.maximum(c, eps)))
    dv_z = np.clip(
        ((dv_s - dv_s.rolling(60, min_periods=1).mean()) / (dv_s.rolling(60, min_periods=1).std() + 1e-4))
        .fillna(0.0).values,
        -5.0, 5.0,
    )

    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    vwap_dev = np.clip((c - ema20) / (np.maximum(c, eps) * vol60), -6.0, 6.0)
    macd_n   = np.clip((ema20 - ema50) / (np.maximum(c, eps) * vol60), -6.0, 6.0)

    gains  = pd.Series(np.maximum(ret1, 0.0)).ewm(span=14, adjust=False).mean().values
    losses = pd.Series(np.maximum(-ret1, 0.0)).ewm(span=14, adjust=False).mean().values + 1e-8
    rsi_n  = (gains / (gains + losses) - 0.5) * 2.0

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        # Unit-safe delta in hours (works identically for ns/us/ms/s DatetimeIndex)
        dt_h = idx.to_series().diff().dt.total_seconds().div(3600.0).fillna(1.0).clip(lower=1e-3).values
        log_dt = np.clip(np.log1p(dt_h), 0.0, 8.0)
        hr, dow = idx.hour.values, idx.dayofweek.values
    else:
        log_dt = np.ones_like(c) * math.log(2.0)
        hr, dow = np.arange(len(c)) % 24, (np.arange(len(c)) // 24) % 7

    feats = np.stack([
        ret_z, hl_rng, co_bdy, up_wk, lo_wk, lv_z, dv_z, vwap_dev,
        np.clip(vol10 / vol60, 0.0, 5.0),
        np.clip(np.log(vol60 + 1e-5) + 5.0, -5.0, 5.0),
        rsi_n, macd_n, log_dt,
        np.sin(2.0 * math.pi * hr / 24.0),
        np.cos(2.0 * math.pi * hr / 24.0),
        np.sin(2.0 * math.pi * dow / 7.0),
    ], axis=-1).astype(np.float32)

    feats = feats[-512:]
    last_close = float(c[-1])
    last_ts_str = str(idx[-1])
    last_ts_utc = parse_utc_ts(idx[-1])
    return feats, last_close, last_ts_str, last_ts_utc


def load_ensemble(ckpt_path: str, device: str):
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    models = []
    for sd in bundle["state_dicts"]:
        m = PlatinumQuant4().to(device).eval()
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        models.append(m)
    return models, bundle.get("seeds", [])


def compute_causal_conviction_series(models, feats: np.ndarray, device: str):
    """Run 5-seed ensemble once and extract full causal trajectory (zero extra compute)."""
    x_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    q_acc, d_acc, j_acc = [], [], []
    with torch.inference_mode():
        for m in models:
            qp, dl, _, jl, _ = m(x_t)
            q_acc.append(qp[0, :, 1, :].cpu().numpy())                    # (L, 5) +5 bar quantiles
            d_acc.append(F.softmax(dl[0, :, :], dim=-1).cpu().numpy())    # (L, 3)
            j_acc.append(torch.sigmoid(jl[0, :]).cpu().numpy())           # (L,)

    q_mean = np.mean(q_acc, axis=0)   # (L, 5)
    d_mean = np.mean(d_acc, axis=0)   # (L, 3)
    j_mean = np.mean(j_acc, axis=0)   # (L,)

    q10_s = q_mean[:, 0]
    q50_s = q_mean[:, 2]
    q90_s = q_mean[:, 4]
    ci_w_s = np.maximum(q90_s - q10_s, 0.25)

    # Exact H5 conviction formula across all L strictly causal steps
    conv_series = 0.55 * (d_mean[:, 2] - d_mean[:, 0]) + 0.45 * np.tanh(q50_s / (0.5 * ci_w_s))

    # Causal rolling z-score over trailing [max(Z_WARMUP, L-Z_WIN) : L]
    valid_hist = conv_series[Z_WARMUP:] if len(conv_series) > Z_WARMUP + 20 else conv_series
    window = valid_hist[-Z_WIN:]
    mu = float(np.mean(window))
    sd = max(float(np.std(window, ddof=1)) if len(window) > 1 else MIN_CONV_STD, MIN_CONV_STD)
    conv_now = float(conv_series[-1])
    conv_z = (conv_now - mu) / sd

    return {
        "q10": float(q10_s[-1]),
        "q50": float(q50_s[-1]),
        "q90": float(q90_s[-1]),
        "p_jump": float(j_mean[-1]),
        "conv": conv_now,
        "conv_z": float(conv_z),
    }


def append_log_rows(log_csv: str, rows: list[dict]):
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if not os.path.exists(log_csv):
        df_new.to_csv(log_csv, index=False)
        return

    # Safely migrate legacy 13-column CSV if needed
    existing_cols = list(pd.read_csv(log_csv, nrows=0).columns)
    if existing_cols == LOG_COLUMNS:
        df_new.to_csv(log_csv, mode="a", header=False, index=False)
    else:
        df_old = pd.read_csv(log_csv)
        for col in LOG_COLUMNS:
            if col not in df_old.columns:
                df_old[col] = np.nan
        df_all = pd.concat([df_old[LOG_COLUMNS], df_new], ignore_index=True)
        df_all.to_csv(log_csv, index=False)


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
    now_dt = pd.Timestamp(datetime.now(timezone.utc))
    now_utc_str = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    today_utc = now_dt.strftime("%Y-%m-%d")
    print(f"=== PlatinumQuant-4.1 Live Shadow Run (v2) | {now_utc_str} | Device: {device} | Gate: {GATE_MODE} ===")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Missing ensemble checkpoint '{CKPT_PATH}'. "
            "Place pq41_ensemble_5seeds_fp16.pt in the repo root or set PQ41_CKPT."
        )

    models, seeds = load_ensemble(CKPT_PATH, device)
    print(f"Loaded {len(models)}-seed ensemble (seeds={seeds})")

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {
            "equity": 1.0,
            "peak_equity": 1.0,
            "daily_date": today_utc,
            "daily_start_equity": 1.0,
            "dd_halted": False,
            "positions": {},
            "last_prices": {},
            "last_ts": {},
            "n_runs": 0,
        }

    # Reset daily drawdown tracker at UTC midnight
    if state.get("daily_date") != today_utc:
        state["daily_date"] = today_utc
        state["daily_start_equity"] = float(state["equity"])
        state["dd_halted"] = False

    # Pass 1: Fetch strictly newer closed bars & realize mark-to-market PnL BEFORE new trades
    fresh_items = []
    realized_port_ret = 0.0

    for tk in ALL_TICKERS:
        cls = TICKER_CLASS[tk]
        df = fetch_and_merge_ohlcv(tk, cls, now_dt)
        if df is None:
            print(f"  [Skip {tk}] no OHLCV data")
            continue

        feats, price, bar_ts_str, bar_ts_utc = build_live_causal_features(df)
        if feats is None or bar_ts_utc is None:
            continue

        prev_ts_utc = parse_utc_ts(state["last_ts"].get(tk))
        if prev_ts_utc is not None and bar_ts_utc <= prev_ts_utc:
            # Strictly monotonic guard: skip closed markets & Yahoo midnight rollbacks
            continue

        prev_pos = float(state["positions"].get(tk, 0.0))
        prev_price = state["last_prices"].get(tk)

        bar_gross_ret = 0.0
        if prev_ts_utc is not None and prev_price is not None and prev_price > 0:
            bar_gross_ret = prev_pos * (price / prev_price - 1.0)
            realized_port_ret += ASSET_WEIGHT * bar_gross_ret

        fresh_items.append({
            "tk": tk,
            "cls": cls,
            "feats": feats,
            "price": price,
            "bar_ts_str": bar_ts_str,
            "prev_pos": prev_pos,
            "bar_gross_ret": bar_gross_ret,
        })

    # Check daily circuit breaker after mark-to-market PnL
    pre_trade_equity = float(state["equity"]) * (1.0 + realized_port_ret)
    daily_start_eq = float(state.get("daily_start_equity", pre_trade_equity))
    daily_ret = pre_trade_equity / max(daily_start_eq, 1e-8) - 1.0
    if daily_ret <= DAILY_DD_STOP:
        state["dd_halted"] = True
    dd_halted = bool(state.get("dd_halted", False))

    # Pass 2: Run ensemble inference ONLY on tickers with a strictly new closed bar
    rows = []
    trade_fee_port_drag = 0.0
    trades_executed = 0

    for item in fresh_items:
        tk = item["tk"]
        cls = item["cls"]
        prev_pos = item["prev_pos"]

        sig = compute_causal_conviction_series(models, item["feats"], device)
        conv = sig["conv"]
        conv_z = sig["conv_z"]
        p_jump = sig["p_jump"]

        if dd_halted:
            # Circuit breaker active: flatten any open position to cash
            new_pos = 0.0
        else:
            if GATE_MODE == "zscore":
                score, ent_thr, ext_thr = conv_z, Z_ENTRY, Z_EXIT
            else:
                score = conv
                ent_thr = RAW_ENTRY_CRYPTO if cls == "Crypto" else RAW_ENTRY_EQUITY
                ext_thr = RAW_EXIT

            new_pos = prev_pos
            if abs(prev_pos) < 1e-6:
                if score > ent_thr:
                    new_pos = 1.0
                elif score < -ent_thr:
                    new_pos = -1.0
            elif prev_pos > 0 and score < ext_thr:
                new_pos = -1.0 if score < -ent_thr else 0.0
            elif prev_pos < 0 and score > -ext_thr:
                new_pos = 1.0 if score > ent_thr else 0.0

            # Hawkes Jump Crash Brake: halve exposure during extreme shock risk
            if p_jump > HAWKES_BRAKE_THR and abs(new_pos) > 0.5:
                new_pos *= 0.5

        dpos = abs(new_pos - prev_pos)
        if dpos > 1e-6:
            trades_executed += 1
        fee_cost = dpos * FEE_RATE
        trade_fee_port_drag += ASSET_WEIGHT * fee_cost
        net_bar_ret = item["bar_gross_ret"] - fee_cost

        state["positions"][tk] = round(new_pos, 4)
        state["last_prices"][tk] = item["price"]
        state["last_ts"][tk] = item["bar_ts_str"]

        rows.append({
            "run_utc": now_utc_str,
            "ticker": tk,
            "asset_class": cls,
            "bar_ts": item["bar_ts_str"],
            "price": round(item["price"], 4),
            "q10": round(sig["q10"], 4),
            "q50": round(sig["q50"], 4),
            "q90": round(sig["q90"], 4),
            "p_jump": round(p_jump, 4),
            "conviction": round(conv, 4),
            "conv_z": round(conv_z, 3),
            "prev_pos": prev_pos,
            "new_pos": new_pos,
            "net_pnl_bps": round(net_bar_ret * 1e4, 2),
            "port_equity": 0.0,  # filled right below once step equity is finalized
        })

    # Update portfolio equity using fixed per-asset weights (1/N)
    step_port_ret = realized_port_ret - trade_fee_port_drag
    if fresh_items:
        state["equity"] = float(state["equity"]) * (1.0 + step_port_ret)
        state["peak_equity"] = max(float(state["peak_equity"]), state["equity"])
        for r in rows:
            r["port_equity"] = round(state["equity"], 6)

    state["n_runs"] = int(state.get("n_runs", 0)) + 1
    dd_pct = (state["equity"] / max(float(state["peak_equity"]), 1e-8) - 1.0) * 100.0
    tot_ret_pct = (state["equity"] - 1.0) * 100.0
    day_ret_pct = (state["equity"] / max(daily_start_eq, 1e-8) - 1.0) * 100.0

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    append_log_rows(LOG_CSV, rows)

    # Active positions summary across the whole portfolio (not just updated bars)
    active_lines = []
    fresh_map = {r["ticker"]: r for r in rows}
    for tk in ALL_TICKERS:
        pos = float(state["positions"].get(tk, 0.0))
        if abs(pos) > 1e-6:
            side = "LONG" if pos > 0 else "SHORT"
            if tk in fresh_map:
                r = fresh_map[tk]
                active_lines.append(
                    f"`{tk}`: {side}({abs(pos):.1f}) z={r['conv_z']:+.2f} conv={r['conviction']:+.3f} jmp={r['p_jump']:.2f}"
                )
            else:
                active_lines.append(f"`{tk}`: {side}({abs(pos):.1f}) (held, market closed)")

    halt_tag = " [DAILY DD STOP]" if dd_halted else ""
    summary = (
        f"*PlatinumQuant-4.1 Live Paper-Trader v2* ({now_utc_str}){halt_tag}\n"
        f"Equity: `{state['equity']:.4f}` (`{tot_ret_pct:+.2f}%` | Day: `{day_ret_pct:+.2f}%` | DD: `{dd_pct:.2f}%`)\n"
        f"Fresh bars: `{len(fresh_items)}/{len(ALL_TICKERS)}` | Trades: `{trades_executed}` | Step: `{step_port_ret*1e4:+.1f} bps`\n"
        f"Active: `{len(active_lines)}/{len(ALL_TICKERS)}` (`{GATE_MODE}`)\n"
        + ("\n".join(active_lines[:10]) if active_lines else "All positions FLAT (cash).")
    )
    print("\n" + summary)
    if fresh_items or not TG_NOTIFY_ONLY_ON_NEW_BARS:
        send_telegram(summary)


if __name__ == "__main__":
    main()
