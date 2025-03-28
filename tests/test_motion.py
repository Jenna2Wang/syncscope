import numpy as np

from syncscope.motion import to_gray, visual_motion


def test_static_frames_have_zero_motion():
    frames = np.ones((10, 8, 8))
    motion = visual_motion(frames)
    assert motion.shape == (10,)
    np.testing.assert_allclose(motion, 0.0)


def test_moving_frames_have_positive_motion():
    rng = np.random.default_rng(0)
    frames = rng.random((12, 16, 16))
    motion = visual_motion(frames)
    assert motion[0] == 0.0
    assert np.all(motion[1:] > 0.0)


def test_roi_restricts_to_region():
    frames = np.zeros((3, 10, 10))
    frames[1:, 0:2, 0:2] = 1.0  # change only the top-left corner
    inside = visual_motion(frames, roi=(0, 2, 0, 2))
    outside = visual_motion(frames, roi=(5, 8, 5, 8))
    assert inside[1] > 0.0
    np.testing.assert_allclose(outside, 0.0)


def test_color_frames_are_reduced_to_gray():
    color = np.zeros((4, 6, 6, 3))
    gray = to_gray(color)
    assert gray.shape == (4, 6, 6)
    # color input should still produce a motion signal of the right length
    assert visual_motion(color).shape == (4,)
