import numpy as np
from flypole.controller import Brain
from flypole.viewer import Viewer


def test_offscreen_frame_has_cartpole_and_status():
    viewer = Viewer(Brain(None), offscreen=True)
    viewer.draw(np.zeros(4), np.zeros(0), 10, 0, "evaluation episode 1/2", 1)
    assert viewer.surface.get_size() == (1200, 650)
    assert viewer.surface.get_at((300, 420))[:3] != (16, 22, 33)
    assert viewer.frames == 1
    viewer.close()
