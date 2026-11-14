"""Kiểm tra SegFormer-B2: cấu hình gốc, kiến trúc, tương đương với code gốc, optimizer, LR, loss, metric, trainer."""

import copy
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.config import load_model_config, strip_notes
from medseg.models import build_segformer, load_mit_pretrained, to_binary_logits
from medseg.paths import ROOT
from medseg.training.evaluate import intersect_and_union, summarize
from medseg.training.losses import segformer_ce_loss
from medseg.training.optim import build_optimizer, build_param_groups, build_scheduler, poly_warmup_lr
from medseg.training.trainer import load_checkpoint, train

ORIGINAL_MIT = ROOT / "base" / "SegFormer-master" / "mmseg" / "models" / "backbones" / "mix_transformer.py"


@pytest.fixture(scope="module")
def cfg():
    return load_model_config("segformer_b2")


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return build_segformer(cfg).eval()


# ----------------------------------------------------------------------------- cấu hình

def test_config_keeps_original_segformer_b2_hparams(cfg):
    assert cfg["backbone"]["depths"] == [3, 4, 6, 3]
    assert cfg["backbone"]["embed_dims"] == [64, 128, 320, 512]
    assert cfg["backbone"]["drop_path_rate"] == 0.1
    assert cfg["decode_head"]["embed_dim"] == 768
    assert cfg["decode_head"]["dropout_ratio"] == 0.1
    opt = cfg["optimizer"]
    assert opt["lr"] == 6e-5 and opt["weight_decay"] == 0.01 and opt["betas"] == [0.9, 0.999]
    assert opt["custom_keys"] == {"pos_block": {"decay_mult": 0.0}, "norm": {"decay_mult": 0.0},
                                  "head": {"lr_mult": 10.0}}
    sch = cfg["lr_schedule"]
    assert (sch["power"], sch["warmup_iters"], sch["warmup_ratio"]) == (1.0, 1500, 1e-6)
    assert cfg["runner"]["max_iters"] == 160000 and cfg["runner"]["batch_size"] == 16
    assert cfg["loss"] == {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_weight": 1.0}


def test_strip_notes_and_unknown_model():
    assert strip_notes({"_a": 1, "b": [{"_c": 2, "d": 3}]}) == {"b": [{"d": 3}]}
    with pytest.raises(KeyError):
        load_model_config("khong_co")


# ----------------------------------------------------------------------------- kiến trúc

def test_output_shapes(model):
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        feats = model.backbone(x)
        logits = model(x)
    assert [tuple(f.shape[1:]) for f in feats] == [(64, 64, 64), (128, 32, 32), (320, 16, 16), (512, 8, 8)]
    assert logits.shape == (2, 2, 64, 64)  # 1/4 độ phân giải
    assert model.predict_logits(x).shape == (2, 2, 256, 256)
    assert model.predict(x).shape == (2, 256, 256)


def test_param_counts(model):
    enc = sum(p.numel() for p in model.backbone.parameters()) / 1e6
    dec = sum(p.numel() for p in model.decode_head.parameters()) / 1e6
    assert 24.0 < enc < 24.5  # paper: MiT-B2 khoảng 24.2M
    assert 3.0 < dec < 3.3


def test_head_state_dict_names_match_mmseg(model):
    keys = set(model.decode_head.state_dict())
    for k in ("linear_c1.proj.weight", "linear_c4.proj.bias", "linear_fuse.conv.weight",
              "linear_fuse.bn.running_mean", "linear_pred.weight", "linear_pred.bias"):
        assert k in keys
    assert "linear_fuse.conv.bias" not in keys  # ConvModule bias='auto' khi có norm


def test_binary_logits_equal_softmax_foreground():
    logits = torch.randn(2, 2, 8, 8)
    b = to_binary_logits(logits, size=(32, 32))
    assert b.shape == (2, 1, 32, 32)
    up = F.interpolate(logits, size=(32, 32), mode="bilinear", align_corners=False)
    assert torch.allclose(torch.sigmoid(b), up.softmax(1)[:, 1:2], atol=1e-6)


def _load_original_mit(monkeypatch):
    """Chạy file mix_transformer.py gốc với các module mmcv/mmseg/timm được thay bằng stub."""

    class DropPath(nn.Module):
        def __init__(self, p=0.0):
            super().__init__()
            self.drop_prob = p

        def forward(self, x):
            return x

    registry = types.SimpleNamespace(register_module=lambda *a, **k: (lambda cls: cls))
    stubs = {
        "timm": types.ModuleType("timm"),
        "timm.models": types.ModuleType("timm.models"),
        "timm.models.layers": types.SimpleNamespace(
            DropPath=DropPath, to_2tuple=lambda v: v if isinstance(v, tuple) else (v, v),
            trunc_normal_=lambda t, std=1.0: nn.init.trunc_normal_(t, std=std)),
        "timm.models.registry": types.SimpleNamespace(register_model=lambda f: f),
        "timm.models.vision_transformer": types.SimpleNamespace(_cfg=lambda **k: k),
        "mmseg": types.ModuleType("mmseg"),
        "mmseg.models": types.ModuleType("mmseg.models"),
        "mmseg.models.builder": types.SimpleNamespace(BACKBONES=registry),
        "mmseg.utils": types.SimpleNamespace(get_root_logger=lambda: None),
        "mmcv": types.ModuleType("mmcv"),
        "mmcv.runner": types.SimpleNamespace(load_checkpoint=None),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    ns = {"__name__": "original_mix_transformer"}
    exec(compile(ORIGINAL_MIT.read_text(encoding="utf-8"), str(ORIGINAL_MIT), "exec"), ns)
    return ns["mit_b2"]


@pytest.mark.skipif(not ORIGINAL_MIT.exists(), reason="không có base/SegFormer-master")
def test_backbone_matches_original_code(model, monkeypatch):
    original = _load_original_mit(monkeypatch)().eval()
    ref = original.state_dict()
    ours = model.backbone.state_dict()
    assert set(ref) == set(ours), set(ref) ^ set(ours)
    for k in ref:
        assert ref[k].shape == ours[k].shape, k
    model.backbone.load_state_dict(ref, strict=True)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        for a, b in zip(original(x), model.backbone(x)):
            assert torch.allclose(a, b, atol=1e-5)
    # LayerNorm: eps 1e-6 trong khối, 1e-5 ở patch embed và attention.sr như bản gốc
    eps = {n: m.eps for n, m in model.backbone.named_modules() if isinstance(m, nn.LayerNorm)}
    eps_ref = {n: m.eps for n, m in original.named_modules() if isinstance(m, nn.LayerNorm)}
    assert eps == eps_ref


def test_load_mit_pretrained(tmp_path, cfg):
    torch.manual_seed(1)
    src = build_segformer(cfg).backbone
    state = {**src.state_dict(), "head.weight": torch.zeros(1000, 512), "head.bias": torch.zeros(1000)}
    torch.save(state, tmp_path / "mit_b2.pth")
    torch.save({"state_dict": {f"backbone.{k}": v for k, v in state.items()}}, tmp_path / "wrapped.pth")
    for name in ("mit_b2.pth", "wrapped.pth"):
        torch.manual_seed(2)
        dst = build_segformer(cfg).backbone
        info = load_mit_pretrained(dst, tmp_path / name)
        assert info["unexpected_keys"] == []
        for k, v in src.state_dict().items():
            assert torch.equal(v, dst.state_dict()[k]), k
    bad = {k: v for k, v in state.items() if not k.startswith("block4")}
    torch.save(bad, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError):
        load_mit_pretrained(build_segformer(cfg).backbone, tmp_path / "bad.pth")
    with pytest.raises(FileNotFoundError):
        load_mit_pretrained(dst, tmp_path / "khong_co.pth")


# ----------------------------------------------------------------------------- optimizer và LR

def test_param_groups_follow_mmcv_custom_keys(model, cfg):
    opt = cfg["optimizer"]
    groups = {g["name"]: g for g in build_param_groups(model, opt["lr"], opt["weight_decay"], opt["custom_keys"])}
    lr, wd = opt["lr"], opt["weight_decay"]
    assert groups["backbone.block1.0.norm1.weight"]["weight_decay"] == 0
    assert groups["backbone.block1.0.attn.norm.bias"]["weight_decay"] == 0
    assert groups["backbone.patch_embed1.norm.weight"]["weight_decay"] == 0
    assert groups["backbone.norm4.weight"]["lr"] == lr
    assert groups["backbone.block2.1.attn.q.weight"]["weight_decay"] == wd
    assert groups["backbone.block2.1.attn.q.bias"]["weight_decay"] == wd
    for name, g in groups.items():
        if name.startswith("decode_head."):
            assert math.isclose(g["lr"], 10 * lr) and g["weight_decay"] == wd, name
    assert not any("pos_block" in n for n in groups)


def test_poly_warmup_schedule(model, cfg):
    max_iters, base = 160000, cfg["optimizer"]["lr"]
    assert math.isclose(poly_warmup_lr(base, 0, max_iters, 1.0, 0.0, 1500, 1e-6), base * 1e-6)
    assert math.isclose(poly_warmup_lr(base, 1500, max_iters, 1.0, 0.0, 1500, 1e-6), base * (1 - 1500 / max_iters))
    assert poly_warmup_lr(base, max_iters, max_iters, 1.0, 0.0, 1500, 1e-6) == 0.0
    mid = poly_warmup_lr(base, 750, max_iters, 1.0, 0.0, 1500, 1e-6)
    assert math.isclose(mid, base * (1 - 750 / max_iters) * (0.5 + 0.5e-6))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    head = next(g for g in optimizer.param_groups if g["name"].startswith("decode_head."))
    assert math.isclose(optimizer.param_groups[0]["lr"], base * 1e-6)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    expect = poly_warmup_lr(base, 10, max_iters, 1.0, 0.0, 1500, 1e-6)
    assert math.isclose(optimizer.param_groups[0]["lr"], expect, rel_tol=1e-9)
    assert math.isclose(head["lr"], 10 * expect, rel_tol=1e-9)


# ----------------------------------------------------------------------------- loss và metric

def test_ce_loss_upsamples_and_matches_cross_entropy():
    logits = torch.randn(2, 2, 16, 16)
    mask = (torch.rand(2, 1, 64, 64) > 0.7).float()
    up = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    expect = F.cross_entropy(up, mask[:, 0].long())
    assert torch.allclose(segformer_ce_loss(logits, mask), expect)


def test_ce_loss_ignored_pixels_count_in_denominator():
    logits = torch.randn(1, 2, 4, 4)
    label = torch.zeros(1, 4, 4)
    label[0, 0] = 255
    full = F.cross_entropy(logits, label.long(), reduction="none", ignore_index=255)
    assert torch.allclose(segformer_ce_loss(logits, label[:, None]), full.sum() / 16)


def test_miou_matches_hand_count():
    pred = torch.tensor([[0, 1, 1, 0]])
    label = torch.tensor([[0, 1, 0, 255]])
    areas = intersect_and_union(pred, label, 2)
    assert areas["intersect"].tolist() == [1, 1] and areas["union"].tolist() == [2, 2]
    m = summarize(areas)
    assert math.isclose(m["mIoU"], 0.5) and math.isclose(m["fg_Dice"], 2 / 3)
    perfect = summarize(intersect_and_union(label.clamp(max=1), label.clamp(max=1), 2))
    assert perfect["mIoU"] == 1.0 and perfect["fg_Dice"] == 1.0


# ----------------------------------------------------------------------------- vòng huấn luyện

def test_train_loop_smoke_and_resume(tmp_path, cfg):
    small = copy.deepcopy(cfg)
    small["runner"].update(max_iters=3, log_interval=1, checkpoint_interval=2, eval_interval=2)
    torch.manual_seed(0)
    batches = [{"image": torch.randn(2, 3, 64, 64), "mask": (torch.rand(2, 1, 64, 64) > 0.8).float()}
               for _ in range(2)]
    model = build_segformer(small)
    best = train(model, batches, batches, small, tmp_path, torch.device("cpu"), log=lambda *_: None)
    for f in ("iter_2.pth", "iter_3.pth", "latest.pth", "best.pth", "log.jsonl"):
        assert (tmp_path / f).exists(), f
    assert best["iter"] in (2, 3) and 0.0 <= best["score"] <= 1.0
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert sum('"loss"' in ln for ln in lines) == 3

    resumed = build_segformer(small)
    assert load_checkpoint(tmp_path / "iter_2.pth", resumed) == 2
    train(resumed, batches, None, small, tmp_path / "resume", torch.device("cpu"),
          resume=tmp_path / "iter_2.pth", log=lambda *_: None)
    assert (tmp_path / "resume" / "iter_3.pth").exists()
    assert not (tmp_path / "resume" / "iter_2.pth").exists()
