"""
상대강도(RS) 대시보드
- S&P 500 + 내 관심종목의 상대강도를 매일 계산해
- docs/index.html (GitHub Pages 대시보드)을 갱신하고
- 디스코드에 상위 종목 요약과 대시보드 링크를 보냅니다.

RS 점수: IBD 방식. 3개월 수익률 40% + 6·9·12개월 수익률 각 20%를 가중합한 뒤
전체 종목 중 백분위(1~99)로 환산. 99면 상위 1%.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
GROUPS = os.environ.get("SCAN_GROUPS", "sp500,watchlist").split(",")
BENCH = os.environ.get("BENCH", "QQQ")                          # 아웃퍼폼 판정 기준 (나스닥 100)
OUTPERF_PERIOD = os.environ.get("OUTPERF_PERIOD", "3m")         # 벤치마크와 비교할 기간 (1w, 1m, 3m, 6m, 12m)
MIN_DV3 = float(os.environ.get("MIN_DV3", "10000000"))          # 유동성: 최근 3거래일 평균 거래대금 $10M 이상
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
TOP_N = int(os.environ.get("TOP_N", "100"))                     # 상위 목록 개수
DISCORD_TOP = int(os.environ.get("DISCORD_TOP", "15"))          # 디스코드에 보낼 상위 종목 수
META_BATCH = int(os.environ.get("META_BATCH", "150"))           # 한 번 실행에 섹터·시총 정보를 새로 받을 최대 종목 수
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
PAGE_URL = os.environ.get("PAGE_URL", "")                       # 비워두면 저장소 이름으로 자동 생성
FORCE = os.environ.get("FORCE", "0") == "1"
# ──────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
DATA.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (rs-dashboard)"}
PERIODS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252}

# 야후 섹터명 → GICS 섹터명 (S&P 500 위키 표기와 통일)
YF_SECTOR = {
    "Technology": "Information Technology", "Healthcare": "Health Care",
    "Financial Services": "Financials", "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples", "Basic Materials": "Materials",
    "Communication Services": "Communication Services", "Industrials": "Industrials",
    "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate",
}
SECTOR_KO = {
    "Information Technology": "정보기술", "Health Care": "헬스케어", "Financials": "금융",
    "Consumer Discretionary": "경기소비재", "Consumer Staples": "필수소비재", "Materials": "소재",
    "Communication Services": "커뮤니케이션", "Industrials": "산업재", "Energy": "에너지",
    "Utilities": "유틸리티", "Real Estate": "부동산", "ETF": "ETF",
}


# ───────────── 종목 목록 ─────────────
def load_sp500() -> pd.DataFrame:
    cache = DATA / "universe_sp500.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400
    if not fresh:
        try:
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                headers=UA, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"], "name": t["Security"], "sector": t["GICS Sector"]})
            if len(df) < 400:
                raise ValueError("종목 수가 비정상적으로 적음")
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] S&P 500 목록 갱신 실패: {e}")
    if not cache.exists():
        return pd.DataFrame(columns=["ticker", "name", "sector"])
    df = pd.read_csv(cache).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    return df[df["ticker"].str.fullmatch(r"[A-Z\-]{1,6}")].drop_duplicates("ticker")


def load_watchlist() -> pd.DataFrame:
    p = ROOT / "watchlist.txt"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            tk, _, name = ln.partition(",")
            tk = re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper())
            rows.append({"ticker": tk, "name": name.strip(), "sector": ""})
    return pd.DataFrame(rows, columns=["ticker", "name", "sector"]).drop_duplicates("ticker")


# ───────────── 시세 ─────────────
def download(tickers: list[str]):
    import yfinance as yf
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="14mo", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                yield t, data[t]
        time.sleep(1)


# ───────────── 섹터·시총 정보 (캐시) ─────────────
def load_meta() -> dict:
    p = DATA / "meta.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def refresh_meta(meta: dict, tickers: list[str]) -> dict:
    """정보가 없거나 7일 지난 종목부터 최대 META_BATCH개만 갱신 (야후 차단 방지)"""
    now = time.time()
    todo = sorted(tickers, key=lambda t: meta.get(t, {}).get("ts", 0))
    todo = [t for t in todo if now - meta.get(t, {}).get("ts", 0) > 7 * 86400][:META_BATCH]
    if not todo:
        return meta
    import yfinance as yf
    ok = 0
    for t in todo:
        try:
            info = yf.Ticker(t).info
            sector = "ETF" if info.get("quoteType") == "ETF" else YF_SECTOR.get(info.get("sector", ""), "")
            meta[t] = {"name": info.get("shortName") or info.get("longName") or "",
                       "sector": sector, "shares": info.get("sharesOutstanding") or 0, "ts": now}
            ok += 1
        except Exception as e:
            print(f"[경고] {t} 정보 조회 실패: {e}")
        time.sleep(0.3)
    print(f"섹터·시총 정보 갱신 {ok}/{len(todo)}종목")
    (DATA / "meta.json").write_text(json.dumps(meta))
    return meta


# ───────────── 계산 ─────────────
def ret(c: pd.Series, n: int) -> float:
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else np.nan


def pct_rank(s: pd.Series) -> pd.Series:
    """백분위 → 1~99 점수"""
    return (s.rank(pct=True) * 98 + 1).round().astype("Int64")


def build_rows(universe: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict, str]:
    tickers = list(universe["ticker"]) + [BENCH, "SPY"]
    info = {r["ticker"]: r for r in universe.to_dict("records")}
    rows, bench = [], {}
    for t, df in download(list(dict.fromkeys(tickers))):
        df = df.dropna(subset=["Close"])
        if len(df) < 70:
            continue
        c, v = df["Close"], df["Volume"].fillna(0)
        r = {k: ret(c, n) for k, n in PERIODS.items()}
        if t in (BENCH, "SPY"):
            bench[t] = r
        if t not in info:                       # 벤치마크 전용 티커는 순위에서 제외
            continue
        u = info[t]
        m = meta.get(t, {})
        last = float(c.iloc[-1])
        s50 = c.rolling(50).mean().iloc[-1]
        s200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
        rows.append({
            "t": t, "n": u["name"] or m.get("name", ""), "g": u["group"],
            "sec": u["sector"] or m.get("sector", "") or "",
            "close": last, "d1": float(c.iloc[-1] / c.iloc[-2] - 1),
            **{f"r{k}": r[k] for k in PERIODS},
            "dv3": float((c * v).tail(3).mean()),
            "mcap": float(m.get("shares", 0) or 0) * last,
            "a50": bool(last > s50), "a200": bool(last > s200) if not np.isnan(s200) else None,
            "hi52": float(last / c.tail(252).max() - 1),
            "date": str(df.index[-1].date()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, bench, ""
    asof = df["date"].mode().iloc[0]

    # IBD 방식 가중 수익률 (데이터가 짧은 종목은 있는 기간만으로)
    w = {"r3m": 2, "r6m": 1, "r9m": 1, "r12m": 1}
    num = sum(df[k].fillna(0) * wt for k, wt in w.items())
    den = sum(df[k].notna() * wt for k, wt in w.items())
    df["score"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    df["rs"] = pct_rank(df["score"])
    for k in ("1w", "1m", "3m"):
        df[f"rs{k}"] = pct_rank(df[f"r{k}"])

    b = bench.get(BENCH, {}).get(OUTPERF_PERIOD, np.nan)
    df["out"] = (df[f"r{OUTPERF_PERIOD}"] > b) & (df["dv3"] >= MIN_DV3) & (df["close"] >= MIN_PRICE)
    return df, bench, asof


def rank_history(df: pd.DataFrame, asof: str) -> dict:
    """상위 목록 순위를 날짜별로 저장하고, 직전 거래일 순위를 돌려줌"""
    hist_dir = DATA / "ranks"
    hist_dir.mkdir(exist_ok=True)
    top = df[df["out"]].sort_values("rs", ascending=False, kind="mergesort").head(TOP_N)
    today_ranks = {t: i + 1 for i, t in enumerate(top["t"])}
    prev_files = sorted(p for p in hist_dir.glob("*.json") if p.stem < asof)
    prev = json.loads(prev_files[-1].read_text()) if prev_files else {}
    (hist_dir / f"{asof}.json").write_text(json.dumps(today_ranks))
    for old in sorted(hist_dir.glob("*.json"))[:-30]:        # 최근 30일치만 보관
        old.unlink()
    return prev


# ───────────── 출력 ─────────────
def page_url() -> str:
    if PAGE_URL:
        return PAGE_URL
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner.lower()}.github.io/{name}/"
    return ""


def write_page(df: pd.DataFrame, bench: dict, asof: str, prev: dict):
    def clean(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 6)
        if isinstance(x, (np.integer,)):
            return int(x)
        if x is pd.NA:
            return None
        return x
    recs = [{k: clean(v) for k, v in r.items()} for r in df.drop(columns=["date"]).to_dict("records")]
    payload = {
        "asof": asof, "generated": dt.datetime.now(NY).strftime("%Y-%m-%d %H:%M ET"),
        "bench": {k: {p: clean(v) for p, v in r.items()} for k, r in bench.items()},
        "benchName": BENCH, "outPeriod": OUTPERF_PERIOD, "minDv3": MIN_DV3, "topN": TOP_N,
        "prev": prev, "sectorKo": SECTOR_KO, "rows": recs,
    }
    tpl = (ROOT / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def send_discord(df: pd.DataFrame, bench: dict, asof: str, prev: dict):
    top = df[df["out"]].sort_values("rs", ascending=False, kind="mergesort").head(TOP_N)
    def pct(x):
        return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.1f}%"
    b, s = bench.get(BENCH, {}), bench.get("SPY", {})
    head = (f"**RS 대시보드 | {asof} 마감**\n"
            f"{BENCH} 3개월 {pct(b.get('3m'))} · SPY 3개월 {pct(s.get('3m'))}\n"
            f"아웃퍼폼+유동성 {int(df['out'].sum())}종목 · RS 90 이상 {int((df['rs'] >= 90).sum())}종목 · 전체 {len(df)}종목")
    lines = [f"{'#':>3} {'CHG':>4} {'TICKER':<7}{'RS':>3}{'1W':>8}{'1M':>8}{'3M':>8}  SECTOR"]
    for i, r in enumerate(top.head(DISCORD_TOP).itertuples(), 1):
        p = prev.get(r.t)
        chg = "NEW" if prev and p is None else ("-" if p is None or p == i else (f"+{p - i}" if p > i else f"{p - i}"))
        lines.append(f"{i:>3} {chg:>4} {r.t:<7}{int(r.rs):>3}{pct(r.r1w):>8}{pct(r.r1m):>8}{pct(r.r3m):>8}  "
                     f"{SECTOR_KO.get(r.sec, r.sec or '-')}")
    new = [t for t in top["t"] if prev and t not in prev]
    msg = head + "\n```\n" + "\n".join(lines) + "\n```"
    if new:
        msg += f"\n상위 {TOP_N} 신규 진입: " + ", ".join(new[:30]) + (" 외" if len(new) > 30 else "")
    url = page_url()
    if url:
        msg += f"\n전체 보기: {url}"
    post(msg[:1990])


def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE:
        print("주말. 종료.")
        return
    sp, wl = load_sp500(), load_watchlist()
    sp["group"], wl["group"] = "sp500", "watchlist"
    universe = pd.concat([sp, wl[~wl["ticker"].isin(sp["ticker"])]], ignore_index=True)
    print(f"대상 {len(universe)}종목 (S&P 500 {len(sp)}, 관심종목 추가분 {len(universe) - len(sp)})")
    if universe.empty:
        post("⚠️ RS 대시보드: 종목 목록을 불러오지 못했습니다. Actions 로그를 확인하세요.")
        return

    meta = refresh_meta(load_meta(), list(universe["ticker"]))
    df, bench, asof = build_rows(universe, meta)
    if df.empty or BENCH not in bench:
        post("⚠️ RS 대시보드: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    if len(df) < len(universe) * 0.5:
        post(f"⚠️ RS 대시보드: 시세를 {len(df)}/{len(universe)}종목만 받았습니다.")

    prev = rank_history(df, asof)
    write_page(df, bench, asof, prev)
    send_discord(df, bench, asof, prev)
    print(f"완료: {asof}, {len(df)}종목, 아웃퍼폼 {int(df['out'].sum())}")


if __name__ == "__main__":
    sys.exit(main())
