# 🥽 ĐỒ ÁN MÔN HỌC: HỆ THỐNG NHẬN DIỆN TRANG BỊ BẢO HỘ LAO ĐỘNG (PPE DETECTION)

> **Môn học:** Thị giác máy tính (Computer Vision) / Học sâu (Deep Learning)  
> **Ứng dụng:** Giám sát an toàn lao động tại công trường xây dựng và nhà xưởng sản xuất  
> **Công nghệ lõi:** Ultralytics YOLO (YOLO26 & YOLO11), RT-DETR (ONNX), OpenCV, Streamlit, PyTorch  

---

## 📌 1. Giới Thiệu Đề Tài (Overview)

Tai nạn lao động tại các công trường xây dựng và khu công nghiệp là vấn đề nhức nhối, trong đó nguyên nhân lớn xuất phát từ việc công nhân không tuân thủ quy định mang đầy đủ **Trang thiết bị bảo hộ cá nhân (Personal Protective Equipment - PPE)** như mũ bảo hộ (`helmet`) và áo phản quang (`vest`).

Mục tiêu của đồ án là xây dựng một **hệ thống nhận diện PPE tự động** có khả năng:
* 🎯 **Phát hiện đồng thời 8 lớp đối tượng:** `boots`, `helmet`, `helmet on`, `no boots`, `no helmet`, `no vest`, `person`, `vest`.
* 🚨 **Cảnh báo vi phạm an toàn:** Tự động phát hiện và khoanh vùng nổi bật các trường hợp vi phạm quy định (`no helmet`, `no vest`).
* ⚡ **Xử lý đa dạng đầu vào:** Ảnh tĩnh, Video tải lên và luồng Camera trực tiếp.
* 🖥️ **Giao diện Demo trực quan:** Xây dựng Web UI tương tác bằng Streamlit, tích hợp lựa chọn linh hoạt giữa nhiều kiến trúc mô hình (YOLO26, YOLO11, RT-DETR).

---

## 📊 2. Kết Quả Huấn Luyện & So Sánh Thực Nghiệm (Model Comparison)

Đồ án tích hợp 3 kiến trúc mô hình tiêu biểu để người dùng và hội đồng đánh giá có thể so sánh trực tiếp:

| Chỉ Số Đánh Giá | 🥇 YOLO26 Nano (`yolo26n.pt`) | 🥈 YOLO11 Small (`yolo11.pt`) | 🥉 RT-DETR (`RTDETR.onnx`) |
| :--- | :---: | :---: | :---: |
| **Kiến trúc mô hình** | CNN (YOLO26 Nano) | CNN (YOLO11 Small) | Vision Transformer (RT-DETR) |
| **Cơ chế hậu xử lý** | NMS | NMS | NMS-Free (Không cần NMS) |
| **Độ chính xác mAP@0.50** | **72.20%** | 70.25% | ~71.50% |
| **Độ phủ (Recall - Bắt lỗi vi phạm)** | **80.67%** | 80.45% | ~79.80% |
| **Kích thước file mô hình** | **5.46 MB** | 19.19 MB | 131.10 MB |
| **Tốc độ suy luận (Inference)** | **~5 ms / frame** | ~15 ms / frame | ~30 ms / frame |
| **Ưu thế nổi bật** | Nhanh nhất, nhẹ nhất, tối ưu real-time | Cân bằng, chuẩn YOLO | Loại bỏ trùng lặp box ở đám đông |

---

## 📁 3. Cấu Trúc Thư Mục Dự Án (Directory Structure)

```text
PPE-Detection/
├── data/
│   ├── raw/                           # Dataset gốc (train / valid / test và data.yaml)
│   ├── preprocessed/                  # Dữ liệu sau tiền xử lý
│   └── process.py                     # Script tiền xử lý và chuẩn hóa nhãn dataset
├── models/                            # Trọng số mô hình đã huấn luyện
│   ├── yolo26n.pt                     # Mô hình YOLO26 Nano (5.46 MB)
│   ├── yolo11.pt                      # Mô hình YOLO11 Small (19.19 MB)
│   └── RTDETR.onnx                    # Mô hình RT-DETR định dạng ONNX (131.1 MB)
├── notebooks/                         # Các Jupyter Notebook nghiên cứu & huấn luyện
│   ├── 01_eda.ipynb                   # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_data_preprocessing.ipynb   # Tiền xử lý dữ liệu và nhãn
│   ├── 03_training_yolo11_kaggle.ipynb # Huấn luyện mô hình YOLO11
│   └── 03_training_yolo26n.ipynb        # Huấn luyện mô hình YOLO26n
├── app.py                             # Giao diện Demo chính (Streamlit Web UI)
├── main.py                            # Script khởi chạy ứng dụng bằng Python
├── requirements.txt                   # Danh sách các thư viện cần thiết
└── README.md                          # Tài liệu báo cáo tổng quan đồ án
```

---

## ✨ 4. Các Tính Năng Nổi Bật Của Giao Diện Demo (`app.py`)

1. **🧠 Lựa chọn Mô hình linh hoạt:** Tùy chọn trực tiếp giữa **YOLO26 Nano**, **YOLO11 Small** hoặc **RT-DETR (ONNX)** trên Sidebar.
2. **🖼️ Nhận diện qua Hình ảnh (Image Detection):**
   - Tải ảnh từ máy tính hoặc **chọn nhanh ảnh mẫu thực tế** có sẵn từ tập `test` để kiểm thử ngay.
   - Thanh trượt điều chỉnh ngưỡng tin cậy (Confidence) và ngưỡng NMS (IoU).
   - Chọn lọc hiển thị các lớp PPE mong muốn.
   - Chế độ so sánh ảnh gốc vs ảnh sau nhận diện (Side-by-side).
   - Thẻ đánh giá an toàn: Thông báo **ĐẠT TIÊU CHUẨN AN TOÀN** (xanh lá) hoặc **CẢNH BÁO VI PHẠM** (đỏ).
   - Tải ảnh kết quả về máy chất lượng cao.
3. **🎬 Nhận diện qua Video (Video Detection):**
   - Hỗ trợ các định dạng video: MP4, AVI, MOV, MKV.
   - Tùy chọn **Frame Skip** giúp tăng tốc độ xử lý video dài.
   - Thanh tiến trình thời gian thực, FPS và màn hình xem trước (Live Preview).
   - Tự động mã hóa chuẩn H.264 tương thích mọi trình duyệt web.
   - Tải video kết quả đã khoanh vùng và gắn bảng cảnh báo.
4. **📷 Kiểm soát an toàn qua Camera (Webcam):**
   - Chụp ảnh trực tiếp từ camera để kiểm tra công nhân trước khi vào cổng công trường.
5. **📊 Bảng So sánh Mô hình (Benchmark Tab):**
   - Hiển thị bảng số liệu so sánh chi tiết và phân tích ưu nhược điểm giữa YOLO26, YOLO11 và RT-DETR.

---

## 🚀 5. Hướng Dẫn Cài Đặt & Khởi Chạy (Quickstart)

### Bước 1: Kích hoạt môi trường ảo (Virtual Environment)
Mở terminal tại thư mục gốc `PPE-Detection`:
```powershell
# Trên Windows:
.\.venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện phụ thuộc (nếu cần cài mới)
```powershell
pip install -r requirements.txt
```

### Bước 3: Khởi chạy Giao diện Demo
Bạn có thể khởi chạy bằng **1 trong 2 lệnh** sau:

- **Cách 1 (Khuyên dùng):** Chạy trực tiếp bằng Streamlit:
  ```powershell
  streamlit run app.py
  ```
- **Cách 2:** Chạy thông qua file `main.py`:
  ```powershell
  python main.py
  ```

Sau khi chạy lệnh, trình duyệt sẽ tự động mở giao diện tại:  
👉 **`http://localhost:8501`**

---

## 👥 6. Thông Tin Đồ Án
- **Đề tài:** Nhận diện trang thiết bị bảo hộ lao động (PPE Detection) bằng Deep Learning.
- **Mục đích:** Đồ án môn học / Báo cáo thực hành Computer Vision.
- **Năm thực hiện:** 2026.
