"""Central configuration: data-lake zones, dataset location and model hyper-parameters."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


@dataclass
class Config:
    # Data lake (local filesystem by default; point GG_LAKE at an s3a:// / hdfs:// mount
    # or an S3-compatible bucket synced locally for the HDFS/S3 deployment option).
    lake: Path = field(default_factory=lambda: _env_path("GG_LAKE", ROOT / "data" / "lake"))
    knowledge_dir: Path = field(default_factory=lambda: _env_path("GG_KNOWLEDGE", ROOT / "knowledge"))
    artifacts: Path = field(default_factory=lambda: _env_path("GG_ARTIFACTS", ROOT / "artifacts"))

    # Elliptic time-step split used by Weber et al. (2019)
    train_max_step: int = 34
    n_steps: int = 49

    # Model
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    lr: float = 5e-3
    weight_decay: float = 5e-4
    epochs: int = 150
    seed: int = 7

    # Anomaly categorisation
    risk_threshold: float = 0.5
    fanout_min_out_degree: int = 8
    fanout_ratio_min: float = 4.0
    bridge_ratio_min: float = 0.5

    # LLM: template (offline) | anthropic | openai
    llm_provider: str = field(default_factory=lambda: os.environ.get("GRAPHGUARD_LLM", "template"))
    llm_model: str = field(default_factory=lambda: os.environ.get("GRAPHGUARD_LLM_MODEL", "claude-sonnet-5-5"))

    @property
    def raw(self) -> Path:
        return self.lake / "raw"

    @property
    def processed(self) -> Path:
        return self.lake / "processed"

    @property
    def analytics(self) -> Path:
        return self.lake / "analytics"

    def ensure(self) -> "Config":
        for p in (self.raw, self.processed, self.analytics, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)
        return self


CFG = Config()
