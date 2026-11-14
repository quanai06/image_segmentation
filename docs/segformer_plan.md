# Kế hoạch áp dụng SegFormer-B2

Phần việc: mô hình đại diện cho nhánh Transformer trong so sánh CNN và Transformer. Code tham khảo: `base/SegFormer-master` (bản chính thức của NVIDIA, dựa trên mmseg 0.x / mmcv 1.x). Thư mục `base/` không có trên Git.

## 1. Cách tiếp cận

Code gốc phụ thuộc mmcv-full 1.2.x và mmseg cũ, không cài được với PyTorch hiện tại. Vì vậy repo **port lại** phần cần thiết sang PyTorch thuần, giữ đúng kiến trúc, tên tham số và hành vi huấn luyện:

| Code gốc | Trong repo |
|---|---|
| `mmseg/models/backbones/mix_transformer.py` (`mit_b2`) | `src/medseg/models/segformer.py`: `MixVisionTransformer` và các khối con |
| `mmseg/models/decode_heads/segformer_head.py` | `src/medseg/models/segformer.py`: `SegFormerHead` |
| `mmseg/models/segmentors/encoder_decoder.py` (`whole_inference`) | `SegFormer.predict_logits`, `SegFormer.predict` |
| `mmseg/models/losses/cross_entropy_loss.py` + `BaseDecodeHead.losses` | `src/medseg/training/losses.py` |
| mmcv `DefaultOptimizerConstructor` (`paramwise_cfg`) | `src/medseg/training/optim.py`: `build_param_groups` |
| mmcv `PolyLrUpdaterHook` + warmup tuyến tính | `src/medseg/training/optim.py`: `poly_warmup_lr`, `build_scheduler` |
| mmseg `mean_iou` | `src/medseg/training/evaluate.py` |
| mmcv `IterBasedRunner`, checkpoint/eval hook | `src/medseg/training/trainer.py` |
| `local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py` | `configs/config.json`, mục `segformer_b2` |

Kiểm chứng: `tests/test_segformer.py` chạy chính file `mix_transformer.py` gốc (thay mmcv/timm bằng stub), so từng khóa của `state_dict` và đầu ra số học của backbone (sai khác < 1e-5). Vì tên khóa trùng khớp, `mit_b2.pth` gốc nạp thẳng vào được.

## 2. Giai đoạn 1 (hiện tại): giữ nguyên siêu tham số gốc

Tất cả nằm trong `configs/config.json`, mục `segformer_b2`.

| Nhóm | Giá trị (như bản gốc) |
|---|---|
| Backbone MiT-B2 | embed_dims 64/128/320/512, heads 1/2/5/8, depths 3/4/6/3, sr_ratios 8/4/2/1, mlp_ratio 4, qkv_bias, drop_path 0.1, LayerNorm eps 1e-6 |
| Pretrained | `pretrained/mit_b2.pth` (ImageNet-1K) |
| Decoder | embed_dim 768, dropout 0.1, align_corners False, BN trong `linear_fuse` |
| Loss | CrossEntropy (softmax, 2 lớp), ignore_index 255, weight 1.0 |
| Optimizer | AdamW, lr 6e-5, betas (0.9, 0.999), weight_decay 0.01; decoder lr ×10; tham số `norm` không weight decay |
| LR | poly power 1.0, min_lr 0, warmup tuyến tính 1500 iter, warmup_ratio 1e-6, cập nhật theo iteration |
| Runner | 160000 iter, batch 16 (gốc: 2 ảnh × 8 GPU), log mỗi 50, checkpoint mỗi 4000, đánh giá mIoU mỗi 16000 |
| Suy luận | `whole`: logits 1/4 → upsample bilinear → softmax → argmax |

**Phần theo pipeline chung của nhóm, không theo bản gốc** (để so sánh công bằng với U-Net):

| | Bản gốc (ADE20K) | Repo |
|---|---|---|
| Kích thước ảnh | crop 512 × 512 | 256 × 256 (resize sẵn) |
| Augmentation | resize ngẫu nhiên 0.5–2.0, crop, lật, PhotoMetricDistortion | preset `fcbformer` |
| Chuẩn hóa | mean/std ImageNet | `imagenet` (cùng giá trị) |
| Số lớp | 150 | 2 (nền, tổn thương) |
| Chia tập | train/val ADE20K | FCBFormer 80/10/10 |

**Bổ sung không đổi siêu tham số:** lưu `best.pth` theo mIoU trên val, log JSON lines, tiếp tục huấn luyện từ checkpoint, hàm `to_binary_logits` đổi logits 2 lớp thành logit 1 kênh `(B, 1, H, W)` cho module metric chung (`softmax[:, 1] == sigmoid(l1 - l0)`).

## 3. Thí nghiệm

| # | Train | Chọn checkpoint | Test | Ghi chú |
|---|---|---|---|---|
| E1 | Kvasir-SEG train | Kvasir val | Kvasir test | thí nghiệm chính |
| E2 | Kvasir-SEG train | Kvasir val | CVC-ClinicDB toàn bộ | chuyển miền, đáng tin nhất |
| E3 | CVC-ClinicDB train | CVC val | CVC test | chỉ để so FCBFormer; val/test rò rỉ khung hình (report mục 7.3) |

`scripts/train_segformer.py` chạy E1 + E2 trong một lệnh (mặc định), và E3 với `--dataset cvc_clinicdb`. Kết quả nằm ở `outputs/segformer_b2/<dataset>/results.json`. Metric đầy đủ (Dice, IoU, HD95, precision, recall, theo nhóm kích thước tổn thương) lấy từ module metric chung khi có.

## 4. Lưu ý trước khi chạy

1. **Thời lượng:** 160k iter × batch 16 trên 800 ảnh Kvasir tương đương 3200 epoch, gấp nhiều lần mức thường dùng cho bộ nhỏ. Giai đoạn 1 vẫn giữ nguyên. Đo tốc độ trước bằng `--max-iters 200` (cột `sec_per_iter` trong `log.jsonl`) để ước tính thời gian trên GPU thực tế.
2. **Tổn thương nhỏ:** logits ở 64 × 64; khoảng 10 ảnh Kvasir và 22 ảnh CVC có tổn thương dưới 50 ô (report mục 7.4). Theo dõi HD95 và recall của nhóm này.
3. **Lệch lớp:** nền chiếm 5.5–9.9 lần tổn thương; CE thuần có thể thiên về nền. Đây là ứng viên đầu tiên cho giai đoạn 2.
4. **Pretrained:** tải `mit_b2.pth` từ link Google Drive/OneDrive trong `base/SegFormer-master/README.md` (mục ImageNet-1K pretrained), đặt vào `pretrained/` (đã bị `.gitignore` chặn qua `*.pth`). Script báo lỗi rõ nếu thiếu, và báo lỗi nếu checkpoint thiếu tham số backbone.

## 5. Giai đoạn 2 (sau khi có kết quả giai đoạn 1)

Chỉ đổi từng yếu tố một, so với baseline giai đoạn 1:
- Loss BCE + Dice (dùng chung với U-Net).
- Số iteration theo epoch (vd. tương đương 200 epoch như FCBFormer) và dừng sớm theo val.
- Ảnh 352 × 352 (`prepare_data.py --size 352`) để tái tạo đúng FCBFormer.
- AMP để tăng tốc trên GPU.

## 6. Lệnh

```bash
# đặt pretrained/mit_b2.pth trước
python scripts/train_segformer.py                                  # E1 + E2
python scripts/train_segformer.py --dataset cvc_clinicdb           # E3
python scripts/train_segformer.py --max-iters 200                  # đo tốc độ
python scripts/train_segformer.py --resume outputs/segformer_b2/kvasir_seg/latest.pth
python scripts/train_segformer.py --eval-only outputs/segformer_b2/kvasir_seg/best.pth
pytest -q tests/test_segformer.py
```
