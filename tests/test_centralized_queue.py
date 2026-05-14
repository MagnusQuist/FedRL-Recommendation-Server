import unittest

from app.ml.centralized import training as centralized_training
from app.ml.centralized.codec import encode_json_blob


def _interaction(reward=1.0):
    return {
        "context": [0.1] * 21,
        "reward": reward,
        "alternative_id": "item-1",
        "nudge_type": "N1",
    }


class CentralizedQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        old_clients_per_round = centralized_training.CLIENTS_PER_ROUND
        centralized_training.CLIENTS_PER_ROUND = 2
        self.addAsyncCleanup(
            setattr,
            centralized_training,
            "CLIENTS_PER_ROUND",
            old_clients_per_round,
        )

    async def test_reupload_replaces_and_second_unique_client_triggers(self):
        service = centralized_training.CentralizedService()
        run_calls = []

        async def fake_run_training_round():
            run_calls.append(sorted(service._pending_uploads))
            service._pending_uploads.clear()

        service._run_training_round = fake_run_training_round
        blob = encode_json_blob([_interaction()])

        version, triggered, queued = await service.process_interactions("client-1", 1, blob)
        self.assertEqual(version, 0)
        self.assertFalse(triggered)
        self.assertEqual(queued, 1)

        _, triggered, queued = await service.process_interactions("client-1", 1, blob)
        self.assertFalse(triggered)
        self.assertEqual(queued, 1)

        _, triggered, queued = await service.process_interactions("client-2", 1, blob)
        self.assertTrue(triggered)
        self.assertEqual(queued, 0)
        self.assertEqual(run_calls, [["client-1", "client-2"]])

    async def test_invalid_context_length_raises_clear_error(self):
        service = centralized_training.CentralizedService()
        invalid = _interaction()
        invalid["context"] = [0.1] * 20

        with self.assertRaisesRegex(ValueError, "context_dim=20, expected 21"):
            await service.process_interactions("client-1", 1, encode_json_blob([invalid]))
