"""
app.ml.modeling.test_v2_modeling
---------------------------------
Unit & Integration Tests for V2 Physics-Informed Multi-Head Neural Architecture.
"""
import unittest

from app.ml.modeling.v2_models import (
    TORCH_AVAILABLE,
    FourierCoordinateEncoder,
    PhysicsInformedDiodeHead,
    SolarV2MultiHeadSurrogate,
    TargetNormalizer,
)

if TORCH_AVAILABLE:
    import torch


class TestV2Modeling(unittest.TestCase):
    """Test suite covering Fourier coordinate encoder, target normalizer, diode head, and V2 surrogate."""

    def test_target_normalizer_invertibility(self):
        """Verify target normalizer Z-Score fit, normalize, and denormalize inverse property."""
        norm = TargetNormalizer()
        targets = [10.0, 20.0, 30.0, 40.0, 50.0]
        norm.fit({"JV": targets})

        y_orig = 25.0
        y_norm = norm.normalize(y_orig, "JV")
        y_recon = norm.denormalize(y_norm, "JV")

        self.assertAlmostEqual(y_orig, y_recon, places=5)

    def test_fourier_encoder_dimensions(self):
        """Verify Fourier encoder maps 1D tensor to (batch_size, 2 * num_freqs)."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        encoder = FourierCoordinateEncoder(num_frequencies=8)
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        encoded = encoder(x)

        self.assertEqual(encoded.shape, (3, 16))

    def test_physics_diode_head_forward(self):
        """Verify PhysicsInformedDiodeHead tensor shapes and forward pass."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        head = PhysicsInformedDiodeHead(in_features=64, fourier_dim=16)
        latent = torch.randn(4, 64)
        v = torch.tensor([[0.0], [0.2], [0.5], [0.8]], dtype=torch.float32)
        fourier_v = torch.randn(4, 16)

        out = head(latent, v, fourier_v)
        self.assertEqual(out.shape, (4, 1))

    def test_v2_surrogate_multi_head_forward(self):
        """Verify SolarV2MultiHeadSurrogate forward pass across JV, QE, and RESPONSIVITY."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        model = SolarV2MultiHeadSurrogate(param_dim=14, num_fourier_freqs=8)
        features = torch.randn(5, 15)  # 14 params + 1 raw x

        out_jv = model(features, stype="JV")
        out_qe = model(features, stype="QE")
        out_resp = model(features, stype="RESPONSIVITY")

        self.assertEqual(out_jv.shape, (5, 1))
        self.assertEqual(out_qe.shape, (5, 1))
        self.assertEqual(out_resp.shape, (5, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
