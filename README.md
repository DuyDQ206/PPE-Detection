# PPE Detection với YOLO11s

Nhận diện trang bị bảo hộ lao động bằng YOLO11s với 4 lớp: **boots**, **helmet**,
**person** và **vest**. Repository gồm các notebook xử lý dữ liệu, huấn luyện trên
Kaggle, kết quả thực nghiệm và code chạy dự đoán trên ảnh, video, webcam.

## Kết quả thực nghiệm

Model được huấn luyện với 3.052 ảnh train và đánh giá trên 222 ảnh validation.

| Model | Params (M) | GFLOPs | Precision | Recall | F1 | mAP50 | mAP50–95 | Latency (ms/ảnh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 9.4143 | 21.4263 | 0.7645 | 0.7997 | 0.7786 | 0.8203 | 0.4937 | 9.9191 |

Các chỉ số P/R/F1/mAP sử dụng thang 0–1. Params và GFLOPs được tính trên model sau
fuse, input 640 × 640. Latency là thời gian inference trung bình mỗi ảnh khi
validation trên Tesla T4, batch 16, FP32; chưa bao gồm đọc/ghi file, preprocess và
postprocess. Chưa có kết quả đánh giá trên tập test độc lập.

Chi tiết cấu hình, kết quả từng lớp và biểu đồ: [Báo cáo YOLO11s](reports/yolo11s/README.md).

### Ví dụ dự đoán trên tập validation

![YOLO11s PPE detection](reports/yolo11s/examples/prediction_01.jpg)

## Cấu trúc dự án

```text
PPE-Detection/
├── configs/                 # Cấu hình dataset
├── models/yolo11s/          # Metadata và vị trí đặt best.pt
├── notebooks/              # EDA, tiền xử lý và huấn luyện
├── reports/yolo11s/         # Metrics, biểu đồ và ảnh dự đoán
├── src/                    # Train, evaluate, predict và các hàm dùng chung
├── main.py                 # Chạy các chức năng từ dòng lệnh
├── requirements.txt        # Thư viện chạy model
└── requirements-notebooks.txt
```

Dataset, trọng số và kết quả chạy local trong `runs/` được Git bỏ qua.

## Cài đặt

Sử dụng Python 3.11 hoặc 3.12. Các lệnh dưới đây chạy bằng PowerShell:

```powershell
git clone --branch feature/yolo11 https://github.com/DuyDQ206/PPE-Detection.git
cd PPE-Detection
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Trên Linux/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`.
Để chạy các notebook, cài thêm:

```powershell
python -m pip install -r requirements-notebooks.txt
```

## Chuẩn bị trọng số

Giải nén ZIP kết quả Kaggle và chép file `ppe_yolo11s/weights/best.pt` vào:

```text
models/yolo11s/best.pt
```

File `.pt` không được lưu trực tiếp trong Git. Có thể kiểm tra checkpoint bằng
SHA256 và đối chiếu với [metadata.json](models/yolo11s/metadata.json):

```powershell
Get-FileHash models/yolo11s/best.pt -Algorithm SHA256
```

## Chạy dự đoán

Chạy từ thư mục gốc dự án sau khi đã đặt trọng số đúng vị trí:

```powershell
# Ảnh, chạy trên CPU
python main.py predict --source "path/to/image.jpg" --device cpu

# Video, chạy trên GPU NVIDIA
python main.py predict --source "path/to/video.mp4" --device 0

# Webcam
python main.py predict --source 0 --show --device 0
```

Confidence mặc định là 0.5, có thể thay bằng `--conf 0.25`. Ảnh/video dự đoán và
nhãn kèm confidence được lưu trong `runs/predict/`.

## Chuẩn bị dữ liệu

Đặt dataset đã tiền xử lý trong `data/preprocessed/`:

```text
data/preprocessed/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/                   # Cần khi đánh giá với --split test
    ├── images/
    └── labels/
```

Nhãn ở định dạng YOLO: `class_id x_center y_center width height`, tọa độ chuẩn hóa
về 0–1. Mapping lớp phải khớp [configs/ppe4.yaml](configs/ppe4.yaml):

| ID | Lớp |
| ---: | --- |
| 0 | boots |
| 1 | helmet |
| 2 | person |
| 3 | vest |

Nếu dataset nằm ở thư mục khác, truyền `--data-root "path/to/dataset"`.
Dùng `--data "path/to/data.yaml"` để chọn YAML riêng; path tương đối trong YAML
được tính theo vị trí file. YAML cần khai báo `train`, `val` và `names`.

## Đánh giá model

```powershell
python main.py evaluate --split val --device cpu
```

Để đánh giá tập test đã chuẩn bị riêng:

```powershell
python main.py evaluate --split test --device 0
```

Kết quả được lưu trong `runs/evaluate/`:

- `metrics_summary.csv`: Params, GFLOPs, Precision, Recall, F1, mAP50, mAP50–95 và latency.
- `summary.json`: kết quả và thông tin môi trường đo.
- `per_class_metrics.csv`: các chỉ số theo từng lớp.
- Confusion matrix và các đường cong đánh giá.

## Huấn luyện

```powershell
python main.py train --epochs 100 --batch 16 --imgsz 640 --device 0
```

Mặc định dùng pretrained `yolo11s.pt`, seed 42 và patience 20. Checkpoint được lưu
trong `runs/train/ppe_yolo11s*/weights/`. Lần train trên Kaggle dừng ở epoch 93;
mAP50–95 validation trong lịch sử train cao nhất tại epoch 73.

Các notebook của dự án:

1. [Khảo sát dữ liệu](notebooks/01_eda.ipynb)
2. [Tiền xử lý dữ liệu](notebooks/02_data_preprocessing.ipynb)
3. [Huấn luyện YOLO11s trên Kaggle](notebooks/03_training_yolo11_kaggle.ipynb)

Xem các tùy chọn bằng `python main.py train --help`, `python main.py evaluate --help`
hoặc `python main.py predict --help`.

## Giới hạn

Model nhận diện 4 lớp hiện có; việc không phát hiện mũ, áo hoặc giày chưa đủ để kết
luận một người vi phạm quy định bảo hộ. Khả năng nhận diện boots còn thấp hơn các
lớp còn lại; cần đánh giá thêm trên dữ liệu camera thực tế trước khi triển khai.
