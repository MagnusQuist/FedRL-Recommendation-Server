import base64
import gzip
import unittest

from app.ml.centralized.codec import decode_json_blob, decode_tuples, encode_json_blob


class CentralizedCodecTests(unittest.TestCase):
    def test_json_blob_round_trip(self):
        payload = {"weights": [1.0, 2.0], "nested": {"ok": True}}

        blob = encode_json_blob(payload)

        self.assertEqual(decode_json_blob(blob), payload)

    def test_invalid_base64_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid base64"):
            decode_json_blob("not base64!")

    def test_invalid_gzip_raises_clear_error(self):
        blob = base64.b64encode(b"plain json but not gzip").decode("utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid gzip"):
            decode_json_blob(blob)

    def test_invalid_json_raises_clear_error(self):
        blob = base64.b64encode(gzip.compress(b"{bad json")).decode("utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            decode_json_blob(blob)

    def test_tuples_payload_must_be_list(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON list"):
            decode_tuples(encode_json_blob({"not": "a list"}))
