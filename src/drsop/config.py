from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, root: Path) -> dict:
    """Make relative paths in the config absolute w.r.t. the repo root."""
    for section in ("data", "model", "train"):
        for key, val in cfg.get(section, {}).items():
            if isinstance(val, str) and any(
                key.endswith(suffix) for suffix in ("_csv", "_dir", "checkpoint", "_repo")
            ):
                cfg[section][key] = str((root / val).resolve())
    return cfg
