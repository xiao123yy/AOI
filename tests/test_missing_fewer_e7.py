"""CPU-only contract tests for the frozen R0+E7 module."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from modules.missing_fewer_e7 import (
    CompositionDescriptor, MissingFewerE7, MissingFewerReference,
)


class MissingFewerE7Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.core = MissingFewerE7()
        self.f16 = torch.randn(5, 384, 24, 24)
        self.f32 = torch.randn(5, 768, 12, 12)

    def test_token_shape_and_fixed_composition(self) -> None:
        tokens = self.core.encode(self.f16, self.f32)
        self.assertEqual(tuple(tokens.shape), (5, 144, 192))
        self.assertEqual(self.core.composition.directions.shape[0], 64)
        self.assertEqual(self.core.composition.centers.numel(), 32)
        self.assertFalse(self.core.composition.directions.requires_grad)

    def test_reference_score_serialization_identity_and_frozen_30a(self) -> None:
        # Keep the production 64x32 constants asserted above.  A 4x4 descriptor
        # makes the covariance smoke test fast enough to remain CPU-only.
        self.core.composition = CompositionDescriptor(4, 4, 31415, .08)
        tokens = self.core.encode(self.f16, self.f32)
        class FastLedoitWolf:
            def fit(self, values):
                self.precision_ = np.eye(values.shape[1], dtype=np.float32)
                return self
        with patch("modules.missing_fewer_e7.LedoitWolf", FastLedoitWolf):
            reference = self.core.build_reference(tokens, "backbone-A", folds=5)
        result_before = self.core.score_tokens(tokens, reference, "backbone-A")
        self.assertTrue(torch.isfinite(result_before["score"]).all())
        self.assertEqual(set(("pred", "node", "relation", "local", "middle", "global", "composition", "e6")),
                         set(result_before).intersection({"pred", "node", "relation", "local", "middle", "global", "composition", "e6"}))
        state_before = {k: v.clone() for k, v in self.core.state_dict().items()}
        self.core.calibrate_threshold(result_before["score"], result_before["score"] + 1, reference, "f1")
        for key, value in self.core.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[key]))
        result_after = self.core.score_tokens(tokens, reference, "backbone-A")
        self.assertTrue(torch.equal(result_before["score"], result_after["score"]))
        with self.assertRaises(RuntimeError):
            self.core.score_tokens(tokens, reference, "backbone-B")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e7_ref.pth"
            reference.save(path)
            restored = MissingFewerReference.load(path)
            restored_score = self.core.score_tokens(tokens, restored, "backbone-A")["score"]
            self.assertTrue(torch.allclose(result_after["score"], restored_score))


if __name__ == "__main__":
    unittest.main()
