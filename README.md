# PPE Detection — YOLO11s

Nhánh `feature/yolo11` chứa phần model YOLO11: notebook đã train trên Kaggle,
checkpoint để chạy local, lệnh train/evaluate/predict và kết quả thực nghiệm.
Model nhận diện **boots, helmet, person, vest** theo đúng thứ tự ID 0–3.

## Kết quả thực tế

Kết quả **validation** trên 222 ảnh, YOLO11s, input 640, Tesla T4:

| Model | Params (M) | GFLOPs | P | R | F1 | mAP50 | mAP50–95 | Latency (ms/ảnh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 9.4143 | 21.4263 | 0.7645 | 0.7997 | 0.7786 | 0.8203 | 0.4937 | 9.9191 |

P/R/F1/mAP dùng thang 0–1. Latency là inference trung bình mỗi ảnh trong validation
batch 16, FP32; chưa phải latency batch 1 hoặc tốc độ toàn ứng dụng.
Nguồn số liệu, kết quả theo lớp, biểu đồ và giới hạn nằm ở
[reports/yolo11s/README.md](reports/yolo11s/README.md).

## Cấu trúc hiện tại

```text
PPE-Detection/
├── configs/ppe4.yaml                    # Cấu hình dataset 4 lớp cho local
├── data/preprocessed/                  # Dataset local, được Git bỏ qua
├── models/yolo11s/
│   ├── best.pt                         # Checkpoint local, được Git bỏ qua
│   ├── metadata.json                   # SHA256, class mapping, nguồn checkpoint
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_training_yolo11_kaggle.ipynb  # Notebook và output Kaggle
├── reports/yolo11s/                     # Metrics, plots, bằng chứng gốc
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── tools/import_yolo11_results.py
├── tests/test_yolo11.py
├── .github/workflows/yolo11.yml
├── main.py
├── requirements.txt
└── requirements-notebooks.txt
```

FastAPI và Streamlit thuộc phần tích hợp hệ thống của dự án. Chưa có code API/UI
trong repo hiện tại; nhánh này hoàn thiện phần model.

## Cài đặt

Chạy từ thư mục gốc repo. Dùng Python 3.11 hoặc 3.12 cho môi trường mới:

```powershell
git clone --branch feature/yolo11 https://github.com/DuyDQ206/PPE-Detection.git
cd PPE-Detection
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Lệnh clone nhánh dùng được sau khi chủ repo push `feature/yolo11` lên GitHub.
Trên Linux/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`.
Nếu dùng CUDA, cài bản PyTorch/torchvision phù hợp GPU và CUDA trước khi cài
requirements. Có thể chạy CPU bằng `--device cpu`.

Để chạy EDA/preprocessing/notebook hoặc tải raw dataset:

```powershell
python -m pip install -r requirements-notebooks.txt
```

## Đưa checkpoint Kaggle vào repo local

Notebook giữ code và output; checkpoint nằm trong ZIP kết quả. Import bằng:

```powershell
python tools/import_yolo11_results.py "E:/download chrome/ppe_yolo11s_results (1).zip" --notebook "E:/download chrome/ppe-yolo11-kaggle.ipynb"
```

Lệnh tạo `models/yolo11s/best.pt`, metadata, báo cáo và bản notebook trong repo.
Số đo gốc được giữ nguyên; các nhãn model cũ YOLO11n được sửa thành YOLO11s dựa trên
cấu hình train. File gốc trong Download được giữ nguyên. Nếu đích đã có checkpoint
khác, lệnh sẽ dừng để tránh ghi đè.

Người clone repo có thể tải `best.pt` từ GitHub Release do chủ repo xuất bản rồi
đặt vào `models/yolo11s/`. Kiểm tra hash bằng:

```powershell
Get-FileHash models/yolo11s/best.pt -Algorithm SHA256
```

So sánh với `sha256` trong [metadata.json](models/yolo11s/metadata.json).
Repo chưa có URL checkpoint cho đến khi chủ repo tạo Release.

## Predict bằng model đã train

Chỉ cần checkpoint, không cần dataset hoặc train lại:

```powershell
python main.py predict --source "path/to/image.jpg" --device cpu
python main.py predict --source "path/to/video.mp4" --device 0
python main.py predict --source 0 --show --device 0
```

Ảnh/video kết quả và nhãn YOLO kèm confidence được lưu dưới `runs/predict/`.
Mặc định confidence 0.5; đổi bằng `--conf 0.25`.
Có thể dùng module: `python -m src.predict --source ...`.
Xem tùy chọn bằng `python main.py predict --help`.

## Dữ liệu để evaluate hoặc train

Dùng đúng dataset 4 lớp của lần train Kaggle:

```text
data/preprocessed/
├── train/images/   + train/labels/
├── valid/images/   + valid/labels/
└── test/images/    + test/labels/   # Chỉ cần khi evaluate --split test
```

ID bắt buộc: `0 boots`, `1 helmet`, `2 person`, `3 vest`.
[configs/ppe4.yaml](configs/ppe4.yaml) đã khai báo mapping này.
Dataset 8 lớp từ workflow cũ cần kiểm tra và chuẩn hóa riêng. Không đổi riêng
`names` để ép khớp checkpoint vì sẽ làm sai ID nhãn.

Dataset ở nơi khác: truyền `--data-root`. Nếu có YAML riêng, truyền `--data`;
path tương đối được tính theo vị trí YAML. `--data-root` hỗ trợ di chuyển path
tuyệt đối cũ từ Kaggle về local. YAML cần cả `train` và `val`; các download hook
trong YAML bị loại bỏ.

## Evaluate và xuất độ đo

```powershell
python main.py evaluate --split val --device cpu
python main.py evaluate --data "path/to/data.yaml" --data-root "path/to/dataset" --split test --device 0
```

Lệnh tạo thư mục mới dưới `runs/evaluate/`, chứa:

- `metrics_summary.csv`: Params, GFLOPs, P, R, F1, mAP50, mAP50–95, latency inference.
- `summary.json`: số đo, split, device, phiên bản thư viện và cấu hình thời gian.
- `per_class_metrics.csv`: P/R/F1/mAP theo lớp; lớp chưa được đánh giá để trống.
- Confusion matrix và các đường cong do Ultralytics xuất.

Báo cáo Kaggle trong `reports/yolo11s/` được giữ làm bằng chứng cho lần train ban đầu.
Test split phải tồn tại; code không tự đổi sang validation.

## Train lại khi cần

```powershell
python main.py train --epochs 100 --batch 16 --imgsz 640 --device 0
```

Mặc định dùng `yolo11s.pt`, seed 42, patience 20, workers 2, tương ứng baseline Kaggle.
Checkpoint mới nằm dưới `runs/train/ppe_yolo11s*/weights/`.
Ultralytics có thể tải pretrained weights ở lần train đầu nếu chưa có trên máy.

## Kiểm tra trước khi push

```powershell
python -m unittest discover -s tests -v
git status --short
```

GitHub Actions chạy cùng kiểm tra offline, không train model hoặc tải checkpoint.

Repo cũ từng chứa khóa Roboflow trong `src/GetData.py`. Code đã chuyển sang biến
môi trường; chủ khóa cần thay/thu hồi khóa cũ vì nó vẫn có thể nằm trong lịch sử Git.
Để tải raw dataset, tạo `.env` từ `.env.example`, điền `ROBOFLOW_API_KEY`, rồi chạy
`python -m src.GetData --output data/raw`. Dataset raw vẫn cần preprocessing.

## Đưa nhánh lên GitHub và chia sẻ trọng số

```powershell
git switch feature/yolo11
git add .
git diff --cached --stat
git commit -m "Integrate trained YOLO11s PPE model and experiment reports"
git push -u origin feature/yolo11
```

`.gitignore` loại dataset, `.pt/.onnx`, outputs trong `runs/`, môi trường ảo và `.env`.

Sau khi push, tạo GitHub Release tag `yolo11s-ppe-v1`, chọn target `feature/yolo11`,
đính kèm `models/yolo11s/best.pt` làm asset. Ghi SHA256 từ metadata trong release notes,
rồi thêm link Release thật vào README. Trọng số chỉ có trên local đến khi bước này xong.

Để nhập phần model vào `main`, mở Pull Request từ `feature/yolo11` sang `main` cho nhóm
review. Phần API/UI có thể dùng `src.utils.load_detector()` để tải checkpoint;
thứ tự lớp và đường dẫn mặc định nằm trong `src/config.py`.
