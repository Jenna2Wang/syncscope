import pytest

from syncscope import io


def test_load_audio_without_extra_explains():
    with pytest.raises(ImportError, match="audio"):
        io.load_audio("nope.wav")


def test_load_video_without_extra_explains():
    with pytest.raises(ImportError, match="video"):
        io.load_video_frames("nope.mp4")
