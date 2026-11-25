"""Single configuration system: base YAML + run profile + command-line overrides.

Why one system: training, serving and monitoring must agree on every setting (image size,
thresholds, registry names). Reading everything through ``load_config`` means a setting can only
come from three places, applied in a fixed order:

1. ``configs/base.yaml``
2. ``configs/profiles/<profile>.yaml`` (deep-merged on top)
3. ``--set key.sub=value`` overrides (applied last)

The chosen profile name is recorded under ``meta.profile``. It is deliberately *not* stored under a
top-level ``profile`` key, because a key that doubles as a config section caused collisions before.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

PROFILES: tuple[str, ...] = ("local_cpu", "colab")
PROFILE_ENV_VAR = "DR_PROFILE"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"

Config = dict[str, Any]


class ConfigError(ValueError):
    """Raised for invalid profiles, malformed overrides or unknown keys."""


def detect_profile() -> str:
    """Return ``colab`` inside Google Colab, otherwise ``local_cpu``.

    Colab is detected without importing ``google.colab`` (importing it outside Colab fails), using
    the module table and the environment variables Colab always sets.
    """
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") or os.environ.get(
        "COLAB_GPU"
    ) is not None:
        return "colab"
    return "local_cpu"


def resolve_profile(explicit: str | None = None) -> str:
    """Pick the profile: explicit argument > ``DR_PROFILE`` env var > auto-detection."""
    profile = explicit or os.environ.get(PROFILE_ENV_VAR) or detect_profile()
    if profile not in PROFILES:
        raise ConfigError(f"Unknown profile {profile!r}; expected one of {PROFILES}.")
    return profile


def _read_yaml(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level.")
    _check_string_keys(data, where=str(path))
    return data


def _check_string_keys(node: Any, where: str, prefix: str = "") -> None:
    """Reject non-string keys.

    Why: YAML 1.1 turns bare ``yes``/``no``/``on``/``off`` keys into booleans, and bare numbers
    into ints. Such keys silently break dotted lookups, so we fail loudly instead.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            if not isinstance(key, str):
                raise ConfigError(
                    f"{where}: key {key!r} under {prefix or '<root>'} is a {type(key).__name__}, "
                    "not a string. Quote it in the YAML file."
                )
            _check_string_keys(value, where, f"{prefix}.{key}" if prefix else key)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge ``override`` into a copy of ``base`` (override wins on leaves)."""
    result: Config = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value`` into (``["a", "b", "c"]``, typed value).

    The value is parsed as a YAML scalar so ``3``, ``0.5``, ``true`` and ``null`` get the expected
    types; key parts always stay strings.
    """
    if "=" not in text:
        raise ConfigError(f"Override {text!r} must look like key.sub=value.")
    key, raw = text.split("=", 1)
    parts = [p.strip() for p in key.strip().split(".")]
    if not parts or any(not p for p in parts):
        raise ConfigError(f"Override {text!r} has an empty key segment.")
    try:
        value = yaml.safe_load(raw) if raw.strip() != "" else ""
    except yaml.YAMLError as exc:  # pragma: no cover - extremely unusual input
        raise ConfigError(f"Cannot parse value in override {text!r}: {exc}") from exc
    return parts, value


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``key.sub=value`` overrides. Unknown keys raise, so typos never pass silently."""
    result = copy.deepcopy(cfg)
    for text in overrides:
        parts, value = parse_override(text)
        node: Any = result
        for i, part in enumerate(parts[:-1]):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Unknown config key {'.'.join(parts[: i + 1])!r} in {text!r}.")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(f"Unknown config key {'.'.join(parts)!r} in {text!r}.")
        if isinstance(node[leaf], dict):
            raise ConfigError(f"{'.'.join(parts)!r} is a section; override one of its keys.")
        node[leaf] = value
    return result


def load_config(
    profile: str | None = None,
    overrides: Iterable[str] = (),
    config_dir: Path | str | None = None,
) -> Config:
    """Load base + profile + overrides and record the profile under ``meta.profile``."""
    cdir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    name = resolve_profile(profile)
    cfg = deep_merge(_read_yaml(cdir / "base.yaml"), _read_yaml(cdir / "profiles" / f"{name}.yaml"))
    cfg = apply_overrides(cfg, overrides)
    cfg["meta"] = {"profile": name}
    return cfg


def get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Dotted lookup, e.g. ``get(cfg, "preprocess.image_size")``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config into ``{"a.b": value}``; used to log every setting to MLflow."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared ``--profile`` and ``--set`` options to a CLI parser."""
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=None,
        help=f"Run profile (default: ${PROFILE_ENV_VAR} or auto-detected).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.batch_size=16. Repeatable.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build the config from parsed ``--profile``/``--set`` arguments."""
    return load_config(profile=args.profile, overrides=args.overrides)


def main(argv: list[str] | None = None) -> int:
    """``python -m dr.config``: print the resolved config, handy for checking overrides."""
    parser = add_config_args(argparse.ArgumentParser(description="Print the resolved config."))
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    print(yaml.safe_dump(cfg, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
