import unittest

from app.ml import aggregation


class AggregationQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_reupload_replaces_and_second_unique_client_triggers(self):
        old_clients_per_round = aggregation.CLIENTS_PER_ROUND
        aggregation.CLIENTS_PER_ROUND = 2
        self.addCleanup(setattr, aggregation, "CLIENTS_PER_ROUND", old_clients_per_round)

        aggregator = aggregation.FLAggregator()
        run_calls = []

        async def fake_run_fedbuff(db):
            run_calls.append(db)
            aggregator._queue.clear()

        aggregator._run_fedbuff = fake_run_fedbuff
        weights = {"backbone.0.bias": [1.0]}

        triggered, queued = await aggregator.enqueue("client-1", 1, 1, weights, db=None)
        self.assertFalse(triggered)
        self.assertEqual(queued, 1)

        triggered, queued = await aggregator.enqueue("client-1", 1, 2, weights, db=None)
        self.assertFalse(triggered)
        self.assertEqual(queued, 1)

        triggered, queued = await aggregator.enqueue("client-2", 1, 1, weights, db=None)
        self.assertTrue(triggered)
        self.assertEqual(queued, 0)
        self.assertEqual(run_calls, [None])
