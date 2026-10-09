# 🥽 Real-Time PPE Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-orange.svg)](https://docs.ultralytics.com/)
[![Backend](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)

> **Hệ thống nhận diện trang thiết bị bảo hộ lao động (PPE) thời gian thực** hỗ trợ giám sát an toàn tại công trường xây dựng và nhà máy.

---

## 📌 Tổng Quan (Overview)

Việc tuân thủ quy định an toàn lao động là yếu tố sống còn tại các công trường. Dự án **PPE Detection System** ứng dụng các mô hình thị giác máy tính tiên tiến (Computer Vision) để tự động phát hiện công nhân có trang bị đầy đủ Mũ bảo hộ (`Helmet`), Áo phản quang (`Vest`), Giày bảo hộ (`Boots`) hay không qua luồng Camera/Video giám sát.

### ✨ Tính Năng Nổi Bật (Features)
* 🎯 **Phát hiện đa lớp PPE:** Nhận diện đồng thời `Helmet`, `NO-Helmet`, `Vest`, `NO-Vest`, `Person`.
* ⚡ **Xử lý thời gian thực (Real-time):** Tối ưu hóa mô hình đạt tốc độ suy luận cao trên video/webcam.
* 🌐 **RESTful API:** Cung cấp API Backend bằng FastAPI hỗ trợ nhận dạng qua file ảnh/video.
* 🖥️ **Giao diện trực quan (Web UI):** Demo tương tác đơn giản, dễ sử dụng được dựng bằng Streamlit.

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

* **Core Models:** YOLOv8, YOLO11, Faster R-CNN / RT-DETR.
* **Backend:** Python, FastAPI, Uvicorn, OpenCV.
* **Frontend:** Streamlit.
* **Data Processing:** PyTorch, Ultralytics, Albumentations, Pandas.

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```text
PPE-Detection/
├── api/                  # FastAPI Backend Server
│   └── main_api.py
├── data/                 # Dataset (Raw & Preprocessed)
│   ├── raw/
│   └── preprocessed/
├── models/               # Chứa các file trọng số (.pt, .onnx)
├── notebooks/            # Jupyter Notebooks (EDA, Preprocessing, Training)
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_training_yolov8.ipynb
├── src/                  # Core modules (Train, Eval, Predict)
│   ├── config.py
│   ├── train.py
│   └── predict.py
├── ui/                   # Streamlit Frontend Web Application
│   └── app_streamlit.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án (Quickstart)

### 1. Yêu cầu môi trường
* Python >= 3.10
* Git
* Card đồ họa NVIDIA (Khuyến khích để huấn luyện mô hình nhanh hơn)

### 2. Cài đặt các thư viện (Installation)
```bash
# Clone repository
git clone [https://github.com/username/PPE-Detection.git](https://github.com/username/PPE-Detection.git)
cd PPE-Detection

# Tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3. Huấn luyện mô hình (Training)
```bash
# Chạy script huấn luyện mô hình YOLOv8
python src/train.py --model yolov8s --epochs 50 --batch 16
```

### 4. Khởi chạy ứng dụng (Run App)
```bash
# Bước 1: Khởi chạy Backend API (Terminal 1)
uvicorn api.main_api:app --reload --port 8000

# Bước 2: Khởi chạy Frontend UI (Terminal 2)
streamlit run ui/app_streamlit.py
```

---

## 📊 Kết Quả Thực Nghiệm (Experiments & Metrics)

Bảng so sánh hiệu năng giữa các kiến trúc mô hình thử nghiệm trên tập dữ liệu Test:

| Model | mAP@0.5 (%) | Recall (%) | Precision (%) | FPS | Model Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s** | 88.5% | 85.2% | 89.1% | 45 | 22.5 MB |
| **YOLO11s** | **91.2%** | **88.0%** | **92.3%** | **60** | **18.4 MB** |
| **Faster R-CNN** | 89.0% | 86.5% | 87.8% | 15 | 108.0 MB |

---

## 👥 Thành Viên Thực Hiện (Contributors)

* **Thành viên 1:** Data Lead & Preprocessing
* **Thành viên 2:** Model Lead & Training (YOLOv8 / YOLO11)
* **Thành viên 3:** System Lead (API, Streamlit UI & Report)

---

## 📝 License

Dự án phục vụ cho mục đích học tập môn **Nhập Môn Thị giác Máy tính (Introduction To Computer Vision)**.