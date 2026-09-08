"""Shared backend constants, validation, and dataclasses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tui.common.paths import PATHS

PROJECT_ROOT = PATHS.data_root
SCRIPT_DIR = PATHS.code_root
RUNTIME_DIR = PATHS.runtime_dir / "vllm"
CONFIG_DIR = PATHS.config_dir / "vllm"
COMPOSE_DIR = PATHS.code_root / "compose" / "vllm"
COMMON_ENV = PATHS.common_env
VLLM_SRC_DIR = PATHS.data_root / ".vllm-src"
DEFAULT_VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"


def validate_name(name: str) -> bool:
    """compose-safe lowercase name (docker compose project names are
    lowercase-only). Also prevents argv/path injection."""
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]*$", name))


@dataclass
class Profile:
    name: str
    container_name: str = ""
    port: str = "8000"
    gpu_id: str = "0"
    tensor_parallel: str = "1"
    config_name: str = ""
    model_id: str = ""
    enable_lora: str = "false"
    max_loras: str = ""
    max_lora_rank: str = ""
    lora_modules: str = ""
    extra_pip_packages: str = ""
    image_tag: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        """Runtime .env path rendered from profiles.yaml."""
        return RUNTIME_DIR / f"{self.name}.env"


@dataclass
class Config:
    name: str
    model: str = ""
    gpu_memory_utilization: str = "0.9"
    extra_params: dict[str, Any] = field(default_factory=dict)
    # Params kept but not passed to the server — stored as comment markers.
    disabled_params: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"{self.name}.yaml"


@dataclass
class ContainerStatus:
    profile_name: str
    container_name: str
    running: bool = False
    status_text: str = "stopped"
    health: str = ""
    port: str = ""
    gpu_id: str = ""
    image: str = ""
    model: str = ""
    lora: bool = False


from tui.common.docker import GpuInfo  # noqa: F401 — re-exported from common


@dataclass
class DockerImage:
    repository: str
    tag: str
    size: str
    created: str
