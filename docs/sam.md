# SAM zero-shot

Đánh giá Segment Anything (Kirillov et al., ICCV 2023) **không train lại** trên Kvasir-SEG và CVC-ClinicDB.

## Chạy

```bash
pip install segment-anything
python scripts/download_sam_weights.py --models vit_b          # vit_b 375 MB, vit_h 2.6 GB
python scripts/run_sam_zeroshot.py --model vit_b --limit 5     # thử nhanh
python scripts/run_sam_zeroshot.py --model vit_b --split test --save-masks
python scripts/summarize_sam.py                                # bảng + hình vào report/
```

Trên Colab (GPU, cho vit_h và P0): `notebooks/sam_colab.ipynb`.

Chạy ở đâu:

| Việc | Thiết bị |
|---|---|
| vit_b, P1–P4 | GPU Mac (MPS) hoặc Colab |
| P0 (`auto_*`), vit_h, chạy toàn bộ dữ liệu | Colab GPU |

Đo trên MacBook với vit_b: P1–P4 khoảng 5 s/ảnh trên MPS (CPU chậm hơn khoảng 2–3 lần); P0 khoảng 36 s/ảnh trên CPU.

## Các kiểu gợi ý

| Mã | Tên trong code | Gợi ý | Chọn mặt nạ |
|---|---|---|---|
| P0 | `auto_top` | không gợi ý (lưới 32 x 32 điểm) | IoU SAM tự dự đoán |
| P0 | `auto_oracle` | không gợi ý | mặt nạ khớp nhãn nhất (dùng nhãn) |
| P1 | `point_top` | 1 điểm tâm mỗi tổn thương | IoU SAM tự dự đoán |
| P1 | `point_oracle` | 1 điểm tâm mỗi tổn thương | mặt nạ khớp nhãn nhất (dùng nhãn) |
| P2 | `points` | 3 điểm dương (gồm điểm tâm) + 3 điểm âm quanh tổn thương | 1 mặt nạ |
| P3 | `box` | hộp bao sát mỗi tổn thương | 1 mặt nạ |
| P4a | `box_jitter_px` | hộp lệch mỗi cạnh tối đa 20 px (như MedSAM) | 1 mặt nạ |
| P4b | `box_jitter_rel` | hộp lệch mỗi cạnh tối đa 10% kích thước hộp | 1 mặt nạ |

Tham số nằm trong `configs/sam.yaml`.

## Quyết định thiết kế

- **Gợi ý lấy từ nhãn** (nhóm A). Đây là cận trên cho kịch bản bác sĩ tương tác, **không** ngang hàng với U-Net/SegFormer vốn không thấy nhãn lúc test. Các kiểu `*_oracle` còn dùng nhãn để chọn mặt nạ.
- **Chạy trên ảnh gốc** (cách dùng chuẩn của SAM), rồi thu mặt nạ về 256 x 256 bằng đúng cách đã dùng cho nhãn (`resize_mask`) và chấm với nhãn trong `data/processed`, để cùng lưới với các mô hình khác.
- **Lọc chấm nhiễu:** bỏ vùng liên thông < 0.1% diện tích ảnh trước khi sinh gợi ý. Mặt nạ JPEG của Kvasir có nhiều chấm nhiễu: không lọc thì 18% ảnh có "nhiều tổn thương", lọc xong còn 4.8%.
- **Nhiều tổn thương:** mỗi vùng một gợi ý, hợp các mặt nạ lại.
- **Tái lập:** seed mỗi ảnh = `seed` + CRC32 của tên ảnh, nên gợi ý ngẫu nhiên giống nhau dù chạy lại hay chạy tiếp.
- **Thiết bị:** tự chọn CUDA, rồi MPS, rồi CPU. Kết quả trên MPS khớp CPU (IoU giữa hai mặt nạ ≥ 0.9995). Riêng P0 dùng `SamAutomaticMaskGenerator` có phép tính float64 mà MPS không hỗ trợ, nên khi mô hình ở MPS thì P0 chạy trên một bản sao ở CPU.

## Đầu ra

```
outputs/sam/<model>/<dataset>_<split>/per_image.csv     # mỗi dòng: ảnh x kiểu gợi ý
outputs/sam/<model>/<dataset>_<split>/masks/<protocol>/<tên>.png   # với --save-masks, 256 x 256
report/tables/sam_zeroshot.csv, sam_by_size.csv
report/figures/sam_prompt_ladder.png, sam_by_size.png, sam_examples_<dataset>.png
```

Cột metric: `dice`, `iou`, `precision`, `recall`, `hd95` (pixel trên lưới 256, NaN nếu dự đoán rỗng), `empty_pred`.

## Dùng SAM với gợi ý từ mô hình khác (nhóm B)

`SamZeroShot.predict` nhận bất kỳ mặt nạ nào làm nguồn gợi ý, nên chỉ cần đưa mặt nạ dự đoán của U-Net/SegFormer (đã phóng về kích thước ảnh gốc) thay cho nhãn:

```python
from medseg.sam import SamZeroShot, load_sam, sam_config

runner = SamZeroShot(load_sam("vit_h"), sam_config())
refined = runner.predict(image, prompt_mask=unet_mask, protocols=["box", "points"])
```

Không dùng các kiểu `*_oracle` và `auto_*` ở nhóm B vì chúng cần nhãn.
