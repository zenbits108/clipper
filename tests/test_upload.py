import json

import pytest

from clipper import upload
from clipper.config import load_channel


class _FakeInsertRequest:
    def __init__(self, video_id):
        self._video_id = video_id

    def next_chunk(self):
        return (None, {"id": self._video_id})


class _FakeThumbnailRequest:
    def __init__(self, recorder, video_id):
        self._recorder = recorder
        self._video_id = video_id

    def execute(self):
        self._recorder["thumbnail_video_id"] = self._video_id
        return {}


class FakeYouTube:
    """Records calls instead of hitting the real API."""

    def __init__(self, video_id="abc123"):
        self.video_id = video_id
        self.calls = {}

    def videos(self):
        outer = self

        class _Videos:
            def insert(self, part, body, media_body):
                outer.calls["insert_part"] = part
                outer.calls["insert_body"] = body
                return _FakeInsertRequest(outer.video_id)

        return _Videos()

    def thumbnails(self):
        outer = self

        class _Thumbnails:
            def set(self, videoId, media_body):
                return _FakeThumbnailRequest(outer.calls, videoId)

        return _Thumbnails()


def test_build_video_body_strips_hashtag_marks_into_tags():
    channel = load_channel("example_channel")
    meta = {
        "title": "A Sharp Reframe",
        "description": "hook line\n\n#shorts #zen",
        "hashtags": ["#shorts", "#zen"],
    }
    body = upload.build_video_body(meta, channel)
    assert body["snippet"]["title"] == "A Sharp Reframe"
    assert body["snippet"]["tags"] == ["shorts", "zen"]
    assert body["snippet"]["categoryId"] == channel.upload.category_id
    assert body["status"]["privacyStatus"] == channel.upload.privacy_status
    assert body["status"]["selfDeclaredMadeForKids"] is False


def test_build_video_body_privacy_override():
    channel = load_channel("example_channel")
    meta = {"title": "t", "description": "d", "hashtags": []}
    body = upload.build_video_body(meta, channel, privacy_status="unlisted")
    assert body["status"]["privacyStatus"] == "unlisted"


def _make_clip_dir(tmp_path, with_thumb=True):
    clip_dir = tmp_path
    (clip_dir / "clip.mp4").write_bytes(b"fake video")
    (clip_dir / "meta.json").write_text(
        json.dumps({"title": "A Sharp Reframe", "description": "desc", "hashtags": ["#shorts"]}),
        encoding="utf-8",
    )
    if with_thumb:
        (clip_dir / "thumb.jpg").write_bytes(b"fake jpg")
    return clip_dir


def test_upload_clip_success_writes_sidecar_and_sets_thumbnail(tmp_path):
    clip_dir = _make_clip_dir(tmp_path)
    channel = load_channel("example_channel")
    fake_youtube = FakeYouTube(video_id="xyz789")

    meta = upload.upload_clip(clip_dir, channel, settings=None, youtube_client=fake_youtube)

    assert meta["video_id"] == "xyz789"
    assert meta["url"] == "https://youtube.com/watch?v=xyz789"
    assert meta["privacy_status"] == channel.upload.privacy_status
    assert fake_youtube.calls["insert_part"] == "snippet,status"
    assert fake_youtube.calls["insert_body"]["snippet"]["title"] == "A Sharp Reframe"
    assert fake_youtube.calls["thumbnail_video_id"] == "xyz789"

    saved = json.loads((clip_dir / "upload.json").read_text())
    assert saved == meta


def test_upload_clip_privacy_override_flows_through(tmp_path):
    clip_dir = _make_clip_dir(tmp_path)
    channel = load_channel("example_channel")
    fake_youtube = FakeYouTube()

    meta = upload.upload_clip(
        clip_dir, channel, settings=None, privacy_status="public", youtube_client=fake_youtube
    )
    assert meta["privacy_status"] == "public"
    assert fake_youtube.calls["insert_body"]["status"]["privacyStatus"] == "public"


def test_upload_clip_missing_thumbnail_still_succeeds(tmp_path):
    clip_dir = _make_clip_dir(tmp_path, with_thumb=False)
    channel = load_channel("example_channel")
    fake_youtube = FakeYouTube()

    meta = upload.upload_clip(clip_dir, channel, settings=None, youtube_client=fake_youtube)
    assert meta["video_id"] == fake_youtube.video_id
    assert "thumbnail_video_id" not in fake_youtube.calls


def test_upload_clip_missing_video_raises(tmp_path):
    channel = load_channel("example_channel")
    (tmp_path / "meta.json").write_text(json.dumps({"title": "t", "description": "d", "hashtags": []}))
    with pytest.raises(upload.UploadError):
        upload.upload_clip(tmp_path, channel, settings=None, youtube_client=FakeYouTube())


def test_upload_clip_missing_meta_raises(tmp_path):
    channel = load_channel("example_channel")
    (tmp_path / "clip.mp4").write_bytes(b"fake")
    with pytest.raises(upload.UploadError):
        upload.upload_clip(tmp_path, channel, settings=None, youtube_client=FakeYouTube())


def test_get_authenticated_service_missing_client_secrets_raises(tmp_path):
    from clipper.config import Settings, OllamaSettings, OpenRouterSettings, YouTubeSettings

    channel = load_channel("example_channel")
    settings = Settings(
        ollama=OllamaSettings(base_url="http://x", default_model="m"),
        openrouter=OpenRouterSettings(base_url="http://x", api_key_env="X", default_model="m"),
        youtube=YouTubeSettings(client_secrets_path=tmp_path / "nope.json", scopes=["scope"]),
        output_dir=tmp_path,
        whisper_model_size="large-v3",
        whisper_device="auto",
        whisper_compute_type="auto",
    )
    with pytest.raises(upload.UploadError):
        upload.get_authenticated_service(channel, settings)
