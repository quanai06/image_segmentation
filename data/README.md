# Thư mục dữ liệu

```
data/
├── raw/                    # dữ liệu gốc, KHÔNG đẩy lên Git
│   ├── archives/           # file nén đã tải (.zip, .rar)
│   ├── kvasir_seg/Kvasir-SEG/{images,masks}/        # 1000 ảnh JPG
│   └── cvc_clinicdb/CVC-ClinicDB/{Original,Ground Truth}/   # 612 ảnh TIFF
├── processed/              # dữ liệu đã chuẩn hóa, KHÔNG đẩy lên Git
│   └── <dataset>/<size>/{images,masks}/<tên>.png
├── splits/                 # file chia tập cố định, CÓ trong Git
│   └── <dataset>/{train,val,test,all}.txt
└── metadata/               # manifest nguồn gốc từng mẫu, CÓ trong Git
    └── <dataset>_manifest.csv
```

- `raw/` và `processed/` tạo lại được bằng các script, nên không đưa lên Git.
- `splits/` và `metadata/` là "hợp đồng" chung của nhóm: ai cũng dùng đúng các file này.
- Mỗi dòng trong `splits/*.txt` là tên mẫu không có đuôi file, khớp với `processed/<dataset>/<size>/images/<tên>.png`.

Tạo lại toàn bộ (chạy từ gốc repo):

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/make_splits.py
python scripts/check_data.py
```
