import unittest

import numpy as np
import pandas as pd

try:
    from app import henan_fusion_upgrade as model
except ModuleNotFoundError:
    import henan_fusion_upgrade as model


class HenanFusionUpgradeTests(unittest.TestCase):
    def test_pre_anchor_is_segment_specific(self):
        rows = pd.DataFrame({
            "period": [1, 12, 20],
            "pre_prediction": [100.0, 100.0, 100.0],
            "anchor_prediction": [200.0, 200.0, 200.0],
        })
        prediction = model.apply_pre_anchor(
            rows,
            {"night": 0.3, "morning": 0.2, "solar_core": 0.0, "evening": 0.1},
        )
        np.testing.assert_allclose(prediction, [130.0, 100.0, 110.0])

    def test_post_strategy_falls_back_when_day_ahead_wins(self):
        rows = pd.DataFrame({
            "period": [1, 1, 2, 2],
            "actual": [100.0, 100.0, -100.0, -100.0],
            "published_da": [100.0, 100.0, 100.0, 100.0],
            "post_prediction": [100.0, 100.0, -200.0, -200.0],
        })
        strategy = model.choose_post_strategy(rows)
        self.assertEqual(strategy["period_rules"]["1"], "da")


if __name__ == "__main__":
    unittest.main()
