import unittest
from dataclasses import dataclass

import numpy as np

from app.ml.fedbuff import aggregate_fedbuff


@dataclass
class Upload:
    backbone_version: int
    interaction_count: int
    weights: dict[str, np.ndarray]


class FedBuffTests(unittest.TestCase):
    def test_single_upload_without_staleness(self):
        current = {"w": np.array([10.0], dtype=np.float32)}
        base = {1: {"w": np.array([10.0], dtype=np.float32)}}
        uploads = [
            Upload(
                backbone_version=1,
                interaction_count=5,
                weights={"w": np.array([8.0], dtype=np.float32)},
            )
        ]

        result = aggregate_fedbuff(current, uploads, base, 1, 1.0, 0.5)

        np.testing.assert_allclose(result["w"], [8.0])

    def test_multiple_uploads_weighted_by_interaction_count(self):
        current = {"w": np.array([10.0], dtype=np.float32)}
        base = {1: {"w": np.array([10.0], dtype=np.float32)}}
        uploads = [
            Upload(1, 1, {"w": np.array([8.0], dtype=np.float32)}),
            Upload(1, 3, {"w": np.array([6.0], dtype=np.float32)}),
        ]

        result = aggregate_fedbuff(current, uploads, base, 1, 1.0, 0.5)

        np.testing.assert_allclose(result["w"], [6.5])

    def test_stale_uploads_are_discounted(self):
        current = {"w": np.array([10.0], dtype=np.float32)}
        base = {
            1: {"w": np.array([10.0], dtype=np.float32)},
            3: {"w": np.array([10.0], dtype=np.float32)},
        }
        uploads = [
            Upload(1, 1, {"w": np.array([4.0], dtype=np.float32)}),
            Upload(3, 1, {"w": np.array([8.0], dtype=np.float32)}),
        ]

        result = aggregate_fedbuff(current, uploads, base, 3, 1.0, 1.0)

        np.testing.assert_allclose(result["w"], [7.0])

    def test_non_positive_weight_raises(self):
        current = {"w": np.array([10.0], dtype=np.float32)}
        base = {1: {"w": np.array([10.0], dtype=np.float32)}}
        uploads = [Upload(1, 0, {"w": np.array([8.0], dtype=np.float32)})]

        with self.assertRaisesRegex(ValueError, "normalizer is non-positive"):
            aggregate_fedbuff(current, uploads, base, 1, 1.0, 0.5)
