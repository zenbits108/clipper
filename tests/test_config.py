import pytest

from clipper.config import ConfigError, load_channel, load_settings


def test_load_settings():
    settings = load_settings()
    assert settings.ollama.base_url
    assert settings.ollama.default_model
    assert settings.openrouter.api_key_env
    assert settings.whisper_model_size
    assert settings.youtube.client_secrets_path.name.endswith(".json")
    assert settings.youtube.scopes


def test_load_example_channel():
    channel = load_channel("example_channel")
    assert channel.name == "example_channel"
    assert channel.selection.provider in {"ollama", "openrouter"}
    assert channel.selection.clip_length.min_seconds < channel.selection.clip_length.max_seconds
    assert channel.reframe.mode in {"face-track", "center", "blur-pillarbox"}
    assert channel.upload.privacy_status in {"private", "unlisted", "public"}
    assert channel.upload.category_id
    assert channel.hashtags


def test_missing_channel_raises():
    with pytest.raises(ConfigError):
        load_channel("does_not_exist")


def test_upload_section_defaults_when_omitted(tmp_path):
    config_dir = tmp_path / "channels"
    config_dir.mkdir()
    (config_dir / "no_upload.yaml").write_text(
        """
name: no_upload
selection:
  prompt: "pick clips"
  provider: ollama
caption:
  font: Anton
  font_size: 96
  primary_color: "&H00FFFFFF"
  highlight_color: "&H0000D7FF"
reframe:
  mode: center
hashtags: []
""",
        encoding="utf-8",
    )
    channel = load_channel("no_upload", config_dir=config_dir)
    assert channel.upload.privacy_status == "private"
    assert channel.upload.category_id == "22"


def test_invalid_privacy_status_raises(tmp_path):
    config_dir = tmp_path / "channels"
    config_dir.mkdir()
    (config_dir / "bad_upload.yaml").write_text(
        """
name: bad_upload
selection:
  prompt: "pick clips"
  provider: ollama
caption:
  font: Anton
  font_size: 96
  primary_color: "&H00FFFFFF"
  highlight_color: "&H0000D7FF"
reframe:
  mode: center
upload:
  privacy_status: super-public
hashtags: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_channel("bad_upload", config_dir=config_dir)
