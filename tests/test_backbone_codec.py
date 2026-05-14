import base64
import gzip
import json
import unittest

import numpy as np

from app.ml.backbone_codec import (
    decode_backbone_arrays,
    decode_backbone_blob,
    encode_backbone_blob,
)


def _blob_from_json_value(value):
    payload = json.dumps(value).encode("utf-8")
    return base64.b64encode(gzip.compress(payload)).decode("utf-8")


class BackboneCodecTests(unittest.TestCase):
    def test_round_trip_blob(self):
        weights = {
            "backbone.0.weight": np.array([[1.0, 2.0]], dtype=np.float32),
            "backbone.0.bias": np.array([3.0], dtype=np.float32),
        }

        blob = encode_backbone_blob(weights)

        self.assertEqual(
            decode_backbone_blob(blob),
            {
                "backbone.0.weight": [[1.0, 2.0]],
                "backbone.0.bias": [3.0],
            },
        )
        arrays = decode_backbone_arrays(blob)
        np.testing.assert_allclose(arrays["backbone.0.weight"], [[1.0, 2.0]])

    def test_invalid_base64_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid base64"):
            decode_backbone_blob("not base64!")

    def test_invalid_gzip_raises_clear_error(self):
        blob = base64.b64encode(b"plain json but not gzip").decode("utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid gzip"):
            decode_backbone_blob(blob)

    def test_invalid_json_raises_clear_error(self):
        blob = base64.b64encode(gzip.compress(b"{bad json")).decode("utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            decode_backbone_blob(blob)

    def test_non_object_payload_raises(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            decode_backbone_blob(_blob_from_json_value([1, 2, 3]))

    def test_non_list_parameter_value_raises(self):
        with self.assertRaisesRegex(ValueError, "must map to a list"):
            decode_backbone_blob(_blob_from_json_value({"weight": 1.0}))
