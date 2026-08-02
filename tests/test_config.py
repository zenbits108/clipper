import pytest

from clipper.config import ConfigError, load_channel, load_settings


def test_load_settings():
    settings = load_settings()
    assert settings.ollama.base_url
    assert settings.ollama.default_model
    assert settings.openrouter.api_key_env
    assert settings.whisper_model_size


def test_load_example_channel():
    channel = load_channel("example_channel")
    assert channel.name == "example_channel"
    assert channel.selection.provider in {"ollama", "openrouter"}
    assert channel.selection.clip_length.min_seconds < channel.selection.clip_length.max_seconds
    assert channel.reframe.mode in {"face-track", "center", "blur-pillarbox"}
    assert channel.hashtags


def test_missing_channel_raises():
    with pytest.raises(ConfigError):
        load_channel("does_not_exist")
