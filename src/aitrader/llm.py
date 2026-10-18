"""Optional LLM filter: ask a model whether to follow each signal ([llm] in the strategy TOML).

Prompts are Python format strings filled with the signal's columns, e.g. {ticker},
{total_value:,.0f}. Answers are cached in the database, so re-running a backtest does not
call the API again. Needs `pip install anthropic` and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime

import pandas as pd


class _SafeDict(defaultdict):
    def __missing__(self, key):
        return "n/a"


def _format(template: str, row: dict) -> str:
    try:
        return template.format_map(_SafeDict(None, row))
    except (ValueError, TypeError):  # e.g. numeric format on a missing value
        clean = {k: (v if v is not None and v == v else "n/a") for k, v in row.items()}
        return template.replace(":,.0f", "").replace(":.1%", "").format_map(_SafeDict(None, clean))


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {"decision": "skip", "reason": f"unparseable answer: {text[:200]}"}


def apply_filter(signals: pd.DataFrame, strategy, con, mode: str = "backtest", log=print) -> pd.DataFrame:
    cfg = strategy.get("llm", {}) or {}
    if not cfg.get("enabled", False) or signals.empty:
        return signals
    cutoff = pd.Timestamp(cfg.get("knowledge_cutoff", "2100-01-01"))
    if mode == "backtest" and not cfg.get("allow_lookahead", False):
        early = signals[pd.to_datetime(signals["known_date"]) <= cutoff]
        if len(early):
            raise RuntimeError(
                f"LLM filter is enabled, but {len(early)} signals are before the model's knowledge cutoff "
                f"({cutoff.date()}). The model may already know how these stocks did, which makes the "
                "backtest look better than reality. Disable [llm], restrict the period to after the cutoff, "
                "or set allow_lookahead = true knowingly."
            )
    import anthropic  # optional dependency

    client = anthropic.Anthropic()
    model = cfg.get("model", "claude-haiku-5-5")
    system = cfg.get("system_prompt", "")
    keep, reasons = [], []
    for row in signals.to_dict("records"):
        user = _format(cfg.get("user_prompt", ""), row)
        key = hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()
        hit = con.execute("SELECT response FROM llm_cache WHERE key = ?", [key]).fetchone()
        if hit:
            text = hit[0]
        else:
            msg = client.messages.create(model=model, max_tokens=int(cfg.get("max_tokens", 300)),
                                         system=system, messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            con.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?)", [key, model, text, datetime.now()])
        ans = _parse(text)
        keep.append(str(ans.get("decision", "skip")).lower() == "buy")
        reasons.append(ans.get("reason", ""))
    out = signals.copy()
    out["llm_reason"] = reasons
    out = out[keep]
    log(f"llm filter: kept {len(out)}/{len(signals)} signals")
    return out
