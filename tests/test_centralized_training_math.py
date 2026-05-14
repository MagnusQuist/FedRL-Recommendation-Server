import math
import unittest

from app.ml.centralized.heads import (
    TSItemHead,
    TSNudgeHead,
    TSPriceHead,
    apply_tuple_to_heads,
)
from app.ml.centralized.models import BackboneEncoder, RewardPredictor, build_optimizer
from app.ml.centralized.trainer import evaluate_backbone_loss, retrain_backbone


def _interaction(reward=1.0, item_id="item-1", nudge_type="N1"):
    return {
        "context": [0.1] * 21,
        "reward": reward,
        "alternative_id": item_id,
        "nudge_type": nudge_type,
    }


class CentralizedTrainingMathTests(unittest.TestCase):
    def test_evaluate_backbone_loss_empty_tuples_is_zero(self):
        self.assertEqual(
            evaluate_backbone_loss(BackboneEncoder(), RewardPredictor(), []),
            0.0,
        )

    def test_retrain_backbone_returns_finite_loss_for_tiny_batch(self):
        backbone = BackboneEncoder()
        reward_predictor = RewardPredictor()
        optimizer = build_optimizer(backbone, reward_predictor)
        tuples = [_interaction(1.0), _interaction(0.0, "item-2", "N2")]

        loss = retrain_backbone(
            backbone,
            reward_predictor,
            optimizer,
            tuples,
            seed=123,
        )

        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)

    def test_apply_tuple_to_heads_updates_all_head_state(self):
        backbone = BackboneEncoder()
        item_head = TSItemHead()
        price_head = TSPriceHead()
        nudge_head = TSNudgeHead()

        apply_tuple_to_heads(backbone, item_head, price_head, nudge_head, _interaction())

        item_state = item_head.state_dict()
        price_state = price_head.state_dict()
        nudge_state = nudge_head.state_dict()
        self.assertIn("item-1", item_state["params"])
        self.assertNotEqual(price_state["A"], 1.0)
        self.assertEqual(nudge_state["interaction_count"], 1)
        self.assertEqual(nudge_state["params"]["N1"]["count"], 1)
