# 🥽 ĐỒ ÁN MÔN HỌC: HỆ THỐNG NHẬN DIỆN TRANG BỊ BẢO HỘ LAO ĐỘNG (PPE DETECTION)

> **Môn học:** Thị giác máy tính (Computer Vision) / Học sâu (Deep Learning)  
> **Ứng dụng:** Giám sát an toàn lao động tại công trường xây dựng và nhà xưởng sản xuất  
> **Công nghệ lõi:** Ultralytics YOLO (YOLO26 & YOLO11), OpenCV, Streamlit, PyTorch  

---

## 📌 1. Giới Thiệu Đề Tài (Overview)

Tai nạn lao động tại các công trường xây dựng và khu công nghiệp là vấn đề nhức nhối, trong đó nguyên nhân lớn xuất phát từ việc công nhân không tuân thủ quy định mang đầy đủ **Trang thiết bị bảo hộ cá nhân (Personal Protective Equipment - PPE)** như mũ bảo hộ (`helmet`) và áo phản quang (`vest`).

Mục tiêu của đồ án là xây dựng một **hệ thống nhận diện PPE tự động** có khả năng:
* 🎯 **Phát hiện đồng thời 8 lớp đối tượng:** `boots`, `helmet`, `helmet on`, `no boots`, `no helmet`, `no vest`, `person`, `vest`.
* 🚨 **Cảnh báo vi phạm an toàn:** Tự động phát hiện và khoanh vùng nổi bật các trường hợp vi phạm quy định (`no helmet`, `no vest`).
* ⚡ **Xử lý thời gian thực:** Đạt tốc độ cao trên ảnh, video và luồng webcam trực tiếp.
* 🖥️ **Giao diện Demo trực quan:** Xây dựng Web UI tương tác thân thiện bằng Streamlit giúp người dùng và hội đồng đánh giá có thể kiểm thử dễ dàng.

---

## 📊 2. Kết Quả Huấn Luyện & So Sánh Thực Nghiệm (Model Comparison)

Trong đồ án này, hai kiến trúc mô hình đã được huấn luyện thực nghiệm qua 100 epochs trên cùng bộ dữ liệu:

| Chỉ Số Đánh Giá | 🥇 YOLO26 Nano (`yolo26n.pt`) | 🥈 YOLO11 Small (`yolo11.pt`) | Nhận Xét & Đánh Giá |
| :--- | :---: | :---: | :--- |
| **Độ chính xác mAP@0.50** | **72.20%** | **70.25%** | **YOLO26n cao hơn +1.95%** |
| **Độ phủ (Recall - Bắt lỗi vi phạm)** | **80.67%** | **80.45%** | **YOLO26n bắt lỗi vi phạm tốt hơn** |
| **Độ chính xác toàn diện mAP50-95** | **44.20%** | **44.20%** | Ngang nhau |
| **Kích thước file mô hình (.pt)** | **5.46 MB** | **19.19 MB** | **YOLO26n nhẹ hơn ~3.5 lần** |
| **Tốc độ suy luận (Inference Speed)** | **~5 ms / frame** | **~15 ms / frame** | **YOLO26n nhanh hơn ~3 lần** |
| **Khả năng xử lý Video/Webcam** | **Rất mượt mà trên CPU/GPU** | Đòi hỏi phần cứng mạnh hơn | **YOLO26n tối ưu cho thời gian thực** |

### 💡 Kết luận lựa chọn mô hình:
* **`yolo26n.pt` được chọn làm mô hình mặc định chính:** Mô hình đạt độ chính xác mAP50 cao hơn (72.2%), độ phủ bắt lỗi vi phạm vượt trội (80.67%), dung lượng siêu nhỏ gọn (5.46MB) và tốc độ suy luận nhanh gấp ~3 lần.
* Trên giao diện Demo, hệ thống vẫn tích hợp cả 2 mô hình để người dùng có thể chuyển đổi so sánh trực quan.

---

## 📁 3. Cấu Trúc Thư Mục Dự Án (Directory Structure)

```text
PPE-Detection/
├── data/
│   ├── raw/                           # Dataset gốc (train / valid / test và data.yaml)
│   ├── preprocessed/                  # Dữ liệu sau tiền xử lý
│   └── process.py                     # Script tiền xử lý và chuẩn hóa nhãn dataset
├── models/                            # Trọng số mô hình đã huấn luyện (.pt)
│   ├── yolo11.pt                      # Mô hình YOLO11 Small (19.19 MB)
│   └── yolo26n.pt                     # Mô hình YOLO26 Nano (5.46 MB - Mô hình chính)
├── notebooks/                         # Các Jupyter Notebook nghiên cứu & huấn luyện
│   ├── 01_eda.ipynb                   # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_data_preprocessing.ipynb   # Tiền xử lý dữ liệu và nhãn
│   ├── 03_training_yolo11_kaggle.ipynb # Huấn luyện mô hình YOLO11
│   └── 03_training_yolo26n.ipynb        # Huấn luyện mô hình YOLO26n
├── app.py                             # Giao diện Demo chính (Streamlit Web UI)
├── main.py                            # Script khởi chạy ứng dụng bằng Python
├── run_ui.bat                         # Phím tắt khởi chạy 1-click trên Windows
├── requirements.txt                   # Danh sách các thư viện cần thiết
└── README.md                          # Tài liệu báo cáo tổng quan đồ án
```

---

## ✨ 4. Các Tính Năng Nổi Bật Của Giao Diện Demo (`app.py`)

1. **🖼️ Nhận diện qua Hình ảnh (Image Detection):**
   - Tải ảnh từ máy tính hoặc **chọn nhanh ảnh mẫu thực tế** có sẵn từ tập `test` để kiểm thử ngay.
   - Thanh trượt điều chỉnh ngưỡng tin cậy (Confidence) và ngưỡng NMS (IoU).
   - Chọn lọc hiển thị các lớp PPE mong muốn.
   - Chế độ so sánh ảnh gốc vs ảnh sau nhận diện (Side-by-side).
   - Hệ thống thẻ trạng thái an toàn: Thông báo **ĐẠT TIÊU CHUẨN AN TOÀN** (xanh lá) hoặc **CẢNH BÁO VI PHẠM** (đỏ).
   - Tải ảnh kết quả về máy chất lượng cao.

2. **🎬 Nhận diện qua Video (Video Detection):**
   - Hỗ trợ các định dạng video: MP4, AVI, MOV, MKV.
   - Tùy chọn **Frame Skip** giúp tăng tốc độ xử lý video dài gấp nhiều lần.
   - Thanh tiến trình thời gian thực, FPS và màn hình xem trước (Live Preview).
   - Tự động mã hóa chuẩn H.264 tương thích mọi trình duyệt web.
   - Tải video kết quả đã khoanh vùng và gắn bảng cảnh báo.

3. **📷 Kiểm soát an toàn qua Camera (Webcam):**
   - Chụp ảnh trực tiếp từ camera để kiểm tra công nhân trước khi vào cổng công trường.

4. **📊 Bảng So sánh Mô hình (Benchmark Tab):**
   - Hiển thị bảng số liệu so sánh chi tiết giữa `yolo26n.pt` và `yolo11.pt`.

---

## 🚀 5. Hướng Dẫn Cài Đặt & Khởi Chạy (Quickstart)

### Bước 1: Kích hoạt môi trường ảo (Virtual Environment)
Mở terminal tại thư mục gốc `PPE-Detection`:
```powershell
# Trên Windows:
.\.venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện (nếu cần cài mới)
```powershell
pip install -r requirements.txt
```

### Bước 3: Khởi chạy Giao diện Demo
Bạn có thể chọn **1 trong 3 cách** cực kỳ đơn giản sau:

- **Cách 1 (Nhanh nhất trên Windows):** Click đúp chuột vào file `run_ui.bat`.
- **Cách 2 (Khuyên dùng):** Chạy lệnh Streamlit từ terminal:
  ```powershell
  streamlit run app.py
  ```
- **Cách 3:** Chạy qua Python:
  ```powershell
  python main.py
  ```

Sau khi khởi chạy, trình duyệt sẽ tự động mở giao diện tại địa chỉ:  
👉 **`http://localhost:8501`**

---

## 👥 6. Thông Tin Đồ Án
- **Đề tài:** Nhận diện trang thiết bị bảo hộ lao động (PPE Detection) bằng Deep Learning.
- **Mục đích:** Đồ án môn học / Báo cáo thực hành Computer Vision.
- **Năm thực hiện:** 2026.
