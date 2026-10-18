"""Loading of global and strategy TOML configs, CLI overrides and sweep variants."""
from __future__ import annotations

import copy
import itertools
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def find_home(start: Path | None = None) -> Path:
    """Repo root: $AITRADER_HOME, or the first parent of cwd that contains config/global.toml."""
    env = os.environ.get("AITRADER_HOME")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "config" / "global.toml").exists():
            return p
    raise FileNotFoundError(
        "config/global.toml not found. Run aitrader from inside the repo or set AITRADER_HOME."
    )


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_dotted(d: dict, key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """'exit.holding_days=126' -> ('exit.holding_days', 126). Values use TOML syntax;
    anything that is not valid TOML is taken as a plain string."""
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key, raw = key.strip(), raw.strip()
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    return key, value


@dataclass
class Settings:
    """Global settings plus resolved paths."""

    home: Path
    raw: dict
    db_override: Path | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return get_dotted(self.raw, key, default)

    def path(self, key: str) -> Path:
        p = Path(self.get(f"paths.{key}"))
        return p if p.is_absolute() else self.home / p

    @property
    def db_path(self) -> Path:
        return self.db_override or self.path("db")

    @property
    def data_end(self) -> str:
        return self.get("data.end") or date.today().isoformat()


@dataclass
class Strategy:
    name: str
    raw: dict
    path: Path | None = None
    variant: str = "base"
    overrides: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return get_dotted(self.raw, key, default)

    @property
    def type(self) -> str:
        return self.get("strategy.type")

    @property
    def status(self) -> str:
        return self.get("strategy.status", "active")

    @property
    def label(self) -> str:
        return self.name if self.variant == "base" else f"{self.name} [{self.variant}]"

    def with_overrides(self, overrides: dict, variant: str | None = None) -> "Strategy":
        raw = copy.deepcopy(self.raw)
        for k, v in overrides.items():
            set_dotted(raw, k, v)
        name = variant if variant is not None else _variant_name(overrides)
        return Strategy(self.name, raw, self.path, name or "base", {**self.overrides, **overrides})

    def variants(self) -> list["Strategy"]:
        """All combinations of the [sweep] table. Without a sweep: just this strategy."""
        sweep = self.get("sweep", {}) or {}
        if not sweep:
            return [self]
        keys = list(sweep.keys())
        values = [v if isinstance(v, list) else [v] for v in sweep.values()]
        return [self.with_overrides(dict(zip(keys, combo))) for combo in itertools.product(*values)]


def _variant_name(overrides: dict) -> str:
    return ",".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())


def load_settings(home: Path | None = None, db: str | None = None) -> Settings:
    home = home or find_home()
    raw = load_toml(home / "config" / "global.toml")
    db_path = Path(db).resolve() if db else None
    return Settings(home=home, raw=raw, db_override=db_path)


def strategy_path(settings: Settings, name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".toml" and p.exists():
        return p
    candidate = settings.home / "config" / "strategies" / f"{name_or_path}.toml"
    if not candidate.exists():
        available = sorted(x.stem for x in (settings.home / "config" / "strategies").glob("*.toml"))
        raise FileNotFoundError(f"Strategy {name_or_path!r} not found. Available: {', '.join(available)}")
    return candidate


def load_strategy(settings: Settings, name_or_path: str, overrides: list[str] | None = None) -> Strategy:
    path = strategy_path(settings, name_or_path)
    raw = load_toml(path)
    name = get_dotted(raw, "strategy.name", path.stem)
    strat = Strategy(name=name, raw=raw, path=path)
    if overrides:
        parsed = dict(parse_override(o) for o in overrides)
        strat = strat.with_overrides(parsed)
    return strat


def list_strategies(settings: Settings) -> list[Strategy]:
    out = []
    for p in sorted((settings.home / "config" / "strategies").glob("*.toml")):
        out.append(load_strategy(settings, str(p)))
    return out
