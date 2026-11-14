"""Tạo pretrained/mit_b2.pth từ bản Hugging Face `nvidia/mit-b2` (ImageNet-1K).

`mit_b2.pth` gốc chỉ có trên Google Drive/OneDrive (README của SegFormer), khó tải tự động.
`nvidia/mit-b2` là cùng checkpoint ImageNet-1K đó, được chuyển sang định dạng `transformers`:
chỉ khác tên khóa, và `attn.kv` bị tách thành `key` + `value`. Script này đổi ngược lại để
`load_mit_pretrained` nạp thẳng vào backbone, rồi kiểm tra đủ khóa và đúng kích thước.

Chạy từ gốc repo:
    python scripts/convert_hf_mit.py                       # -> pretrained/mit_b2.pth
    python scripts/convert_hf_mit.py --repo nvidia/mit-b0 --out pretrained/mit_b0.pth --model ...
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_model_config  # noqa: E402
from medseg.models import build_segformer, load_mit_pretrained  # noqa: E402
from medseg.paths import from_root, to_root  # noqa: E402

# tên trong transformers (sau tiền tố block.{i}.{j}.) -> tên trong mix_transformer.py gốc
BLOCK_RENAMES = {
    "layer_norm_1": "norm1",
    "layer_norm_2": "norm2",
    "attention.self.query": "attn.q",
    "attention.self.sr": "attn.sr",
    "attention.self.layer_norm": "attn.norm",
    "attention.output.dense": "attn.proj",
    "mlp.dense1": "mlp.fc1",
    "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
    "mlp.dense2": "mlp.fc2",
}


def download_hf(repo: str, filename: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"tải {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})) as r, \
            open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def convert(hf_state: dict) -> dict:
    out, kv = {}, {}
    for k, v in hf_state.items():
        k = k.removeprefix("segformer.").removeprefix("encoder.")
        if k.startswith("classifier."):
            continue  # lớp phân loại ImageNet, load_mit_pretrained cũng bỏ qua
        if m := re.fullmatch(r"patch_embeddings\.(\d)\.(proj|layer_norm)\.(weight|bias)", k):
            i, part, p = m.groups()
            out[f"patch_embed{int(i) + 1}.{'proj' if part == 'proj' else 'norm'}.{p}"] = v
        elif m := re.fullmatch(r"layer_norm\.(\d)\.(weight|bias)", k):
            out[f"norm{int(m[1]) + 1}.{m[2]}"] = v
        elif m := re.fullmatch(r"block\.(\d)\.(\d+)\.(.+)\.(weight|bias)", k):
            i, j, name, p = m.groups()
            prefix = f"block{int(i) + 1}.{j}"
            if name in ("attention.self.key", "attention.self.value"):
                kv.setdefault(f"{prefix}.attn.kv.{p}", {})[name.rsplit(".", 1)[1]] = v
            elif name in BLOCK_RENAMES:
                out[f"{prefix}.{BLOCK_RENAMES[name]}.{p}"] = v
            else:
                raise KeyError(f"Không biết đổi tên khóa: {k}")
        else:
            raise KeyError(f"Không biết đổi tên khóa: {k}")
    # kv gốc là một Linear(dim, 2*dim): nửa đầu là key, nửa sau là value
    for name, parts in kv.items():
        out[name] = torch.cat([parts["key"], parts["value"]], dim=0)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="nvidia/mit-b2")
    p.add_argument("--model", default="segformer_b2", help="mục trong configs/config.json để kiểm tra")
    p.add_argument("--out", default="pretrained/mit_b2.pth")
    p.add_argument("--cache", default="pretrained/hf_cache")
    args = p.parse_args()

    src = download_hf(args.repo, "pytorch_model.bin", from_root(args.cache) / args.repo.replace("/", "__") / "pytorch_model.bin")
    state = convert(torch.load(src, map_location="cpu", weights_only=True))

    model = build_segformer(load_model_config(args.model))
    expected = model.backbone.state_dict()
    extra = sorted(set(state) - set(expected))
    bad_shape = [k for k in state if k in expected and state[k].shape != expected[k].shape]
    if extra or bad_shape:
        raise RuntimeError(f"Khóa thừa: {extra[:5]}; sai kích thước: {bad_shape[:5]}")

    out = from_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    info = load_mit_pretrained(model.backbone, out)  # báo lỗi nếu thiếu tham số backbone
    print(f"đã ghi {to_root(out)}: {info['loaded']} tensor, khớp toàn bộ backbone {args.model}")


if __name__ == "__main__":
    main()
