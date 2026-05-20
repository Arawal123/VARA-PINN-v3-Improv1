from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics.kovasznay import KovasznayFlow, center_pressure


def test_kovasznay_shapes_and_pressure_centering():
    flow = KovasznayFlow(reynolds=40.0)
    _, _, xy = flow.grid(8, 7)
    exact = flow.exact_np(xy)
    assert exact["u"].shape == (56, 1)
    assert exact["v"].shape == (56, 1)
    assert exact["p"].shape == (56, 1)
    assert abs(float(center_pressure(exact["p"]).mean())) < 1e-12
    assert np.isfinite(exact["omega"]).all()

