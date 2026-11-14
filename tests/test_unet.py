"""Kiểm tra hợp đồng đầu ra và khả năng huấn luyện, không cần dữ liệu tải về."""

import unittest

import torch
from torch import nn

from medseg.models import UNet


class TestUNet(unittest.TestCase):
    def test_default_model_at_pipeline_resolution(self):
        model = UNet().eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_odd_rectangular_input_and_custom_channels(self):
        model = UNet(in_channels=1, out_channels=2, base_channels=4)
        logits = model(torch.randn(2, 1, 33, 49))
        self.assertEqual(tuple(logits.shape), (2, 2, 33, 49))

    def test_backward_reaches_every_parameter(self):
        torch.manual_seed(42)
        model = UNet(base_channels=4)
        images = torch.randn(2, 3, 32, 32)
        masks = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = nn.BCEWithLogitsLoss()(model(images), masks)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(model.enc1.layers[0].weight.grad.abs().sum().item(), 0)

    def test_head_returns_raw_logits(self):
        model = UNet(base_channels=4).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-2)
            logits = model(torch.randn(1, 3, 16, 16))
        torch.testing.assert_close(logits, torch.full_like(logits, -2))

    def test_invalid_input(self):
        model = UNet(base_channels=4)
        for shape in [(3, 32, 32), (1, 1, 32, 32), (1, 3, 15, 32)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))


if __name__ == "__main__":
    unittest.main()
