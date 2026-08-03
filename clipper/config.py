"""Loaders for global settings (config/settings.yaml) and per-channel
profiles (config/channels/<name>.yaml). Nothing channel- or provider-specific
is hardcoded outside these YAML files.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


class ConfigError(Exception):
    """Raised when a settings or channel config file is missing or malformed."""


@dataclasses.dataclass
class OllamaSettings:
    base_url: str
    default_model: str


@dataclasses.dataclass
class OpenRouterSettings:
    base_url: str
    api_key_env: str
    default_model: str


@dataclasses.dataclass
class YouTubeSettings:
    client_secrets_path: Path
    scopes: list


@dataclasses.dataclass
class Settings:
    ollama: OllamaSettings
    openrouter: OpenRouterSettings
    youtube: YouTubeSettings
    output_dir: Path
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str


@dataclasses.dataclass
class ClipLength:
    min_seconds: float
    max_seconds: float


@dataclasses.dataclass
class SelectionConfig:
    prompt: str
    tone: str
    clip_length: ClipLength
    max_candidates: int
    provider: str
    model: Optional[str]


@dataclasses.dataclass
class CaptionConfig:
    font: str
    font_size: int
    primary_color: str
    highlight_color: str
    position: str


@dataclasses.dataclass
class ReframeConfig:
    mode: str


@dataclasses.dataclass
class UploadConfig:
    privacy_status: str
    category_id: str


@dataclasses.dataclass
class ChannelConfig:
    name: str
    selection: SelectionConfig
    caption: CaptionConfig
    reframe: ReframeConfig
    upload: UploadConfig
    hashtags: list


VALID_REFRAME_MODES = {"face-track", "center", "blur-pillarbox"}
VALID_PROVIDERS = {"ollama", "openrouter"}
VALID_CAPTION_POSITIONS = {"bottom_safe", "middle", "top_safe"}
VALID_PRIVACY_STATUSES = {"private", "unlisted", "public"}


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise ConfigError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def load_settings(path: Optional[Path] = None) -> Settings:
    path = path or DEFAULT_CONFIG_DIR / "settings.yaml"
    if not path.exists():
        raise ConfigError(f"Settings file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    llm_raw = _require(raw, "llm", str(path))
    ollama_raw = _require(llm_raw, "ollama", "llm.ollama")
    openrouter_raw = _require(llm_raw, "openrouter", "llm.openrouter")
    youtube_raw = raw.get("youtube", {})
    paths_raw = raw.get("paths", {})
    transcribe_raw = raw.get("transcribe", {})

    return Settings(
        ollama=OllamaSettings(
            base_url=_require(ollama_raw, "base_url", "llm.ollama"),
            default_model=_require(ollama_raw, "default_model", "llm.ollama"),
        ),
        openrouter=OpenRouterSettings(
            base_url=_require(openrouter_raw, "base_url", "llm.openrouter"),
            api_key_env=_require(openrouter_raw, "api_key_env", "llm.openrouter"),
            default_model=_require(openrouter_raw, "default_model", "llm.openrouter"),
        ),
        youtube=YouTubeSettings(
            client_secrets_path=REPO_ROOT / youtube_raw.get(
                "client_secrets_path", "config/youtube_client_secret.json"
            ),
            scopes=list(
                youtube_raw.get("scopes", ["https://www.googleapis.com/auth/youtube"])
            ),
        ),
        output_dir=REPO_ROOT / paths_raw.get("output_dir", "output"),
        whisper_model_size=transcribe_raw.get("model_size", "large-v3"),
        whisper_device=transcribe_raw.get("device", "auto"),
        whisper_compute_type=transcribe_raw.get("compute_type", "auto"),
    )


def load_channel(name: str, config_dir: Optional[Path] = None) -> ChannelConfig:
    config_dir = config_dir or DEFAULT_CONFIG_DIR / "channels"
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in config_dir.glob("*.yaml")) if config_dir.exists() else []
        raise ConfigError(
            f"No channel config found for '{name}' at {path}. "
            f"Available channels: {', '.join(available) or '(none)'}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    sel_raw = _require(raw, "selection", str(path))
    clip_len_raw = sel_raw.get("clip_length", {})
    caption_raw = _require(raw, "caption", str(path))
    reframe_raw = _require(raw, "reframe", str(path))

    provider = _require(sel_raw, "provider", f"selection ({path})")
    if provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"Invalid selection.provider '{provider}' in {path}; must be one of {sorted(VALID_PROVIDERS)}"
        )

    reframe_mode = _require(reframe_raw, "mode", f"reframe ({path})")
    if reframe_mode not in VALID_REFRAME_MODES:
        raise ConfigError(
            f"Invalid reframe.mode '{reframe_mode}' in {path}; must be one of {sorted(VALID_REFRAME_MODES)}"
        )

    selection = SelectionConfig(
        prompt=_require(sel_raw, "prompt", f"selection ({path})"),
        tone=sel_raw.get("tone", ""),
        clip_length=ClipLength(
            min_seconds=float(clip_len_raw.get("min_seconds", 20)),
            max_seconds=float(clip_len_raw.get("max_seconds", 90)),
        ),
        max_candidates=int(sel_raw.get("max_candidates", 8)),
        provider=provider,
        model=sel_raw.get("model"),
    )
    caption_position = caption_raw.get("position", "bottom_safe")
    if caption_position not in VALID_CAPTION_POSITIONS:
        raise ConfigError(
            f"Invalid caption.position '{caption_position}' in {path}; "
            f"must be one of {sorted(VALID_CAPTION_POSITIONS)}"
        )

    caption = CaptionConfig(
        font=_require(caption_raw, "font", f"caption ({path})"),
        font_size=int(_require(caption_raw, "font_size", f"caption ({path})")),
        primary_color=_require(caption_raw, "primary_color", f"caption ({path})"),
        highlight_color=_require(caption_raw, "highlight_color", f"caption ({path})"),
        position=caption_position,
    )
    reframe = ReframeConfig(mode=reframe_mode)

    upload_raw = raw.get("upload", {})
    privacy_status = upload_raw.get("privacy_status", "private")
    if privacy_status not in VALID_PRIVACY_STATUSES:
        raise ConfigError(
            f"Invalid upload.privacy_status '{privacy_status}' in {path}; "
            f"must be one of {sorted(VALID_PRIVACY_STATUSES)}"
        )
    upload = UploadConfig(
        privacy_status=privacy_status,
        category_id=str(upload_raw.get("category_id", "22")),  # 22 = People & Blogs
    )

    return ChannelConfig(
        name=raw.get("name", name),
        selection=selection,
        caption=caption,
        reframe=reframe,
        upload=upload,
        hashtags=list(raw.get("hashtags", [])),
    )
