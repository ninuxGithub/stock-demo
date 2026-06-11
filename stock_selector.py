import datetime
import re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from akshare.stock.stock_hot_search_baidu import stock_hot_search_baidu
from akshare.stock.stock_hot_up_em import stock_hot_up_em
from akshare.stock.stock_zh_a_sina import stock_zh_a_daily, stock_zh_a_spot


def normalize_stock_code(code: str) -> str:
    code = str(code).strip()
    if not code:
        return code
    code = code.upper()
    if code.startswith("SH") or code.startswith("SZ") or code.startswith("BJ"):
        return code.lower()
    digits = re.search(r"(\d{6})", code)
    if digits:
        code6 = digits.group(1)
        if code6.startswith(("6", "9", "5")):
            return f"sh{code6}"
        return f"sz{code6}"
    return code.lower()


def is_excluded_market(code: str) -> bool:
    code = normalize_stock_code(code)
    if len(code) < 8:
        return False
    digits = code[-6:]
    return digits.startswith(("300", "301", "688"))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def fetch_a_share_spot() -> pd.DataFrame:
    df = stock_zh_a_spot()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).map(normalize_stock_code)
    df = df[df["代码"].str.match(r"^(sh|sz|bj)\d{6}$", na=False)]
    df = df[~df["代码"].map(is_excluded_market)]
    df["成交额"] = safe_numeric(df["成交额"])
    df["成交量"] = safe_numeric(df["成交量"])
    df["最新价"] = safe_numeric(df["最新价"])
    df["涨跌幅"] = safe_numeric(df["涨跌幅"])
    return df.sort_values(by="成交额", ascending=False).reset_index(drop=True)


def extract_code_from_label(label: str) -> Optional[str]:
    if not isinstance(label, str):
        return None
    match = re.search(r"([A-Z]{2})?(\d{6})", label.upper())
    if match:
        return normalize_stock_code(match.group(0))
    return None


def fetch_hot_codes() -> Set[str]:
    hot_codes: Set[str] = set()
    today = datetime.date.today().strftime("%Y%m%d")
    try:
        df = stock_hot_search_baidu(symbol="A股", date=today, time="今日")
        for label in df.get("名称/代码", []):
            code = extract_code_from_label(label)
            if code and not is_excluded_market(code):
                hot_codes.add(code)
    except Exception:
        pass
    try:
        df = stock_hot_up_em()
        for code in df.get("代码", []):
            if isinstance(code, str):
                normalized = normalize_stock_code(code)
                if normalized and not is_excluded_market(normalized):
                    hot_codes.add(normalized)
    except Exception:
        pass
    return hot_codes


def calculate_kline_trend(symbol: str, lookback_days: int = 60) -> Optional[Dict[str, object]]:
    try:
        end_date = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=lookback_days * 2)).strftime("%Y%m%d")
        df = stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end_date, adjust="qfq")
    except Exception:
        return None
    if df.empty or len(df) < 20:
        return None
    df = df.sort_values(by="date").reset_index(drop=True)
    df["close"] = safe_numeric(df["close"])
    df["open"] = safe_numeric(df["open"])
    df["high"] = safe_numeric(df["high"])
    df["low"] = safe_numeric(df["low"])
    df["volume"] = safe_numeric(df["volume"])
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    current = df.iloc[-1]
    prior = df.iloc[-2]
    score = 0.0
    strong_trend = current["close"] > current["ma5"] > current["ma10"] > current["ma20"]
    follow_trend = current["close"] > current["ma10"] > current["ma20"]
    score += 0.35 if strong_trend else 0.15 if follow_trend else 0.0
    score += 0.15 if current["close"] > current["ma20"] else 0.0
    recent_gain = (current["close"] / df.iloc[-6]["close"] - 1) if len(df) >= 6 else 0.0
    score += min(max(recent_gain / 0.1, 0.0), 1.0) * 0.2
    avg_vol_5 = df["volume"].rolling(5).mean().iloc[-1]
    if avg_vol_5 > 0:
        score += min(max(current["volume"] / avg_vol_5 - 1.0, 0.0), 1.0) * 0.15
    strong_bull = current["close"] > current["open"] and current["close"] > prior["close"]
    score += 0.1 if strong_bull else 0.0
    labels: List[str] = []
    if strong_trend:
        labels.append("5/10/20日均线多头")
    elif follow_trend:
        labels.append("价格保持多头态势")
    if current["close"] > current["ma20"]:
        labels.append("站上20日均线")
    if current["close"] > prior["close"]:
        labels.append("连续上涨")
    if current["volume"] > avg_vol_5:
        labels.append("量能放大")
    return {
        "symbol": symbol,
        "close": float(current["close"]),
        "open": float(current["open"]),
        "percent": float((current["close"] / prior["close"] - 1) * 100 if prior["close"] else 0.0),
        "ma5": float(current["ma5"]),
        "ma10": float(current["ma10"]),
        "ma20": float(current["ma20"]),
        "ma60": float(current["ma60"]),
        "score": float(score),
        "trend_labels": labels,
        "dates": df["date"].tolist(),
    }


def classify_trend(score: float, hot: bool) -> str:
    if score >= 0.7:
        return "强势看涨"
    if score >= 0.55:
        return "看涨"
    if score >= 0.45:
        return "观望"
    return "暂避"


def build_stock_score(
    row: pd.Series,
    hot_codes: Set[str],
    history: Dict[str, object],
) -> Dict[str, object]:
    hot_signal = row["代码"] in hot_codes
    hot_score = 1.0 if hot_signal else 0.0
    technical_score = history["score"] if history else 0.0
    combined_score = min(1.0, technical_score * 0.8 + hot_score * 0.2)
    return {
        "代码": row["代码"],
        "名称": row["名称"],
        "最新价": row["最新价"],
        "涨跌幅": row["涨跌幅"],
        "成交额": row["成交额"],
        "热度": hot_signal,
        "技术评分": round(technical_score, 3),
        "消息评分": round(hot_score, 3),
        "综合评分": round(combined_score, 3),
        "趋势判断": classify_trend(combined_score, hot_signal),
        "趋势标签": ";".join(history["trend_labels"]) if history else "",
    }


def select_stocks(
    top_n: int = 30,
    universe_n: int = 200,
    fa_filter: bool = True,
) -> pd.DataFrame:
    spot_df = fetch_a_share_spot()
    if fa_filter:
        spot_df = spot_df[spot_df["成交额"] > 1e7]
    spot_df = spot_df.head(universe_n)
    hot_codes = fetch_hot_codes()
    rows: List[Dict[str, object]] = []
    for idx, row in spot_df.iterrows():
        symbol = row["代码"]
        history = calculate_kline_trend(symbol)
        if history is None:
            continue
        rows.append(build_stock_score(row, hot_codes, history))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(by=["综合评分", "技术评分", "成交额"], ascending=False)
    return result.head(top_n).reset_index(drop=True)


def main() -> None:
    picks = select_stocks(top_n=30, universe_n=120)
    if picks.empty:
        print("未能获取选股结果，请检查网络或日期设置。")
        return
    print("已选出 A 股看涨候选股票：")
    print(picks[["代码", "名称", "最新价", "涨跌幅", "成交额", "趋势判断", "技术评分", "消息评分", "趋势标签"]].to_string(index=False))


if __name__ == "__main__":
    main()
