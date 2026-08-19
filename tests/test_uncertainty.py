import numpy as np
import unittest

from src.uncertainty import calibrate_mc_dropout, finite_sample_quantile


class CalibrationTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_higher_order_statistic(self):
        scores = np.arange(1.0, 21.0)
        self.assertEqual(finite_sample_quantile(scores, alpha=0.10), 20.0)

    def test_mc_dropout_calibration_is_per_output(self):
        y_pred = np.zeros((20, 2))
        y_std = np.ones((20, 2))
        y_true = np.column_stack([np.ones(20), np.full(20, 3.0)])
        scales = calibrate_mc_dropout(y_true, y_pred, y_std, alpha=0.10)
        np.testing.assert_allclose(scales, [1.0, 3.0])

    def test_mc_dropout_calibration_validates_shapes(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            calibrate_mc_dropout(
                np.zeros((2, 2)), np.zeros((2, 1)), np.ones((2, 2))
            )


if __name__ == "__main__":
    unittest.main()
