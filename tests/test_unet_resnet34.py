"""Kiểm tra U-Net ResNet34 không cần Internet hoặc dữ liệu tải về."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from medseg.models import UNetResNet34


class TestUNetResNet34(unittest.TestCase):
    def test_pipeline_resolution(self):
        model = UNetResNet34(pretrained=False).eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_odd_rectangular_inputs_and_custom_output_channels(self):
        model = UNetResNet34(pretrained=False, out_channels=2).eval()
        for height, width in [(65, 97), (255, 257), (32, 33)]:
            with self.subTest(size=(height, width)), torch.no_grad():
                logits = model(torch.randn(1, 3, height, width))
                self.assertEqual(tuple(logits.shape), (1, 2, height, width))

    def test_backward_finetunes_encoder_and_decoder(self):
        torch.manual_seed(42)
        model = UNetResNet34(pretrained=False)
        images = torch.randn(2, 3, 64, 64)
        masks = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = nn.BCEWithLogitsLoss()(model(images), masks)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertGreater(model.encoder.conv1.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_default_requests_imagenet_and_preserves_encoder_weights(self):
        # Mô phỏng backbone đã nạp trọng số để phát hiện việc khởi tạo đè.
        backbone = resnet34(weights=None)
        with torch.no_grad():
            backbone.conv1.weight.fill_(0.125)
            backbone.bn1.running_mean.fill_(0.25)
        with patch("medseg.models.unet_resnet34.resnet34", return_value=backbone) as factory:
            model = UNetResNet34()
        factory.assert_called_once_with(weights=ResNet34_Weights.IMAGENET1K_V1)
        torch.testing.assert_close(
            model.encoder.conv1.weight, torch.full_like(model.encoder.conv1.weight, 0.125)
        )
        torch.testing.assert_close(
            model.encoder.bn1.running_mean, torch.full_like(model.encoder.bn1.running_mean, 0.25)
        )
        self.assertIsInstance(model.encoder.fc, nn.Identity)
        self.assertIsInstance(model.encoder.avgpool, nn.Identity)

    def test_returns_raw_logits(self):
        model = UNetResNet34(pretrained=False).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-2)
            logits = model(torch.randn(1, 3, 32, 32))
        torch.testing.assert_close(logits, torch.full_like(logits, -2))

    def test_invalid_input(self):
        model = UNetResNet34(pretrained=False)
        for shape in [(3, 64, 64), (1, 1, 64, 64), (1, 3, 31, 64)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))

    def test_invalid_configuration(self):
        for kwargs in [{"out_channels": 0}, {"out_channels": True}, {"pretrained": "yes"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                UNetResNet34(**kwargs)


if __name__ == "__main__":
    unittest.main()
