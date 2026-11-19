import os
import sys
import time
import tempfile
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# CẤU HÌNH TRANG VÀ GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Nhận Diện PPE - Đồ Án Môn Học",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho giao diện hiện đại & học thuật
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        color: #777;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .alert-danger-box, .alert-safe-box {
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    .alert-danger-box { background: #fff0f0; color: #a52626; border: 1px solid #e3b0b0; }
    .alert-safe-box { background: #f0f7f0; color: #27632c; border: 1px solid #b7d1b7; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 0.9rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TÌM KIẾM ĐƯỜNG DẪN MÔ HÌNH THÔNG MINH
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

def find_model_file(filename):
    cands = [
        BASE_DIR / "models" / filename,
        BASE_DIR / "notebooks" / filename,
        Path("models") / filename,
        Path("notebooks") / filename,
        Path(filename),
    ]
    for p in cands:
        if p.is_file():
            return str(p)
    return None

YOLO26N_PATH = find_model_file("yolo26n.pt")
YOLO11_PATH = find_model_file("yolo11.pt")
RTDETR_PATH = find_model_file("RTDETR.onnx")

AVAILABLE_MODELS = {}
if YOLO26N_PATH:
    AVAILABLE_MODELS["YOLO26 Nano (yolo26n.pt)"] = {
        "path": YOLO26N_PATH,
        "type": "PyTorch (.pt)",
        "arch": "CNN (YOLO26 Nano)",
        "mAP50": "72.2%",
        "recall": "80.7%",
        "precision": "59.4%",
        "size": "5.46 MB",
        "speed": "~5 ms / frame (Cực nhanh)",
    }
if YOLO11_PATH:
    AVAILABLE_MODELS["YOLO11 Small (yolo11.pt)"] = {
        "path": YOLO11_PATH,
        "type": "PyTorch (.pt)",
        "arch": "CNN (YOLO11 Small)",
        "mAP50": "70.2%",
        "recall": "80.4%",
        "precision": "60.9%",
        "size": "19.19 MB",
        "speed": "~15 ms / frame (Trung bình)",
    }
if RTDETR_PATH:
    AVAILABLE_MODELS["RT-DETR (RTDETR.onnx)"] = {
        "path": RTDETR_PATH,
        "type": "ONNX Runtime (.onnx)",
        "arch": "Vision Transformer (RT-DETR)",
        "mAP50": "~71.5%",
        "recall": "~79.8%",
        "precision": "~62.1%",
        "size": "131.1 MB",
        "speed": "~30 ms / frame (NMS-Free Transformer)",
    }

if not AVAILABLE_MODELS:
    st.error(" Không tìm thấy các file trọng số `yolo26n.pt`, `yolo11.pt` hoặc `RTDETR.onnx` trong thư mục `models/`!")
    st.stop()

# ---------------------------------------------------------
# BẢNG MÀU VÀ DỊCH NGHĨA CLASS PPE
# ---------------------------------------------------------
CLASS_INFO = {
    "boots": {"vn": "Giày bảo hộ", "color": (255, 165, 0), "is_violation": False},       # BGR: Cam
    "helmet": {"vn": "Mũ bảo hộ", "color": (0, 200, 0), "is_violation": False},         # BGR: Xanh lá
    "helmet on": {"vn": "Đang đội mũ", "color": (0, 200, 0), "is_violation": False},     # BGR: Xanh lá
    "no boots": {"vn": "Không giày bảo hộ", "color": (0, 140, 255), "is_violation": True}, # BGR: Cam đậm
    "no helmet": {"vn": "KHÔNG ĐỘI MŨ", "color": (0, 0, 240), "is_violation": True},     # BGR: Đỏ đậm
    "no vest": {"vn": "KHÔNG ÁO PHẢN QUANG", "color": (0, 0, 240), "is_violation": True}, # BGR: Đỏ đậm
    "person": {"vn": "Người / Công nhân", "color": (255, 144, 30), "is_violation": False}, # BGR: Xanh dương
    "vest": {"vn": "Áo phản quang", "color": (0, 220, 220), "is_violation": False},      # BGR: Vàng sáng
}

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ---------------------------------------------------------
# HÀM VẼ BOUNDING BOX TÙY BIẾN CHO AN TOÀN LAO ĐỘNG
# ---------------------------------------------------------
def draw_ppe_boxes(image: np.ndarray, results, selected_classes=None, show_conf=True):
    """
    Vẽ bounding box với màu sắc quy ước an toàn lao động:
    - An toàn (helmet, vest, boots, person): Xanh lá / Xanh dương / Vàng
    - Vi phạm (no helmet, no vest, no boots): Đỏ cảnh báo nổi bật
    """
    img_draw = image.copy()
    h, w = img_draw.shape[:2]
    
    detection_counts = {}
    violations = []
    
    if len(results) == 0 or results[0].boxes is None:
        return img_draw, detection_counts, violations

    boxes = results[0].boxes
    names = results[0].names

    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id))
        conf = float(box.conf[0].item())
        
        if selected_classes and cls_name not in selected_classes:
            continue
            
        detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
        
        info = CLASS_INFO.get(cls_name, {"vn": cls_name, "color": (128, 128, 128), "is_violation": False})
        color = info["color"]
        is_violation = info["is_violation"]
        
        if is_violation:
            violations.append((cls_name, conf))
        
        # Tọa độ box
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Độ dày đường vẽ: vi phạm viền dày hơn
        thickness = 3 if is_violation else 2
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Text nhãn
        vn_label = info["vn"]
        label_text = f"{vn_label} {conf:.2f}" if show_conf else vn_label
        if is_violation:
            label_text = f" {label_text}"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Nền cho nhãn
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = y1
        cv2.rectangle(img_draw, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
        
        # Màu chữ: Trắng hoặc đen
        text_color = (255, 255, 255) if is_violation or color[1] < 150 else (0, 0, 0)
        cv2.putText(img_draw, label_text, (x1 + 3, y1 - baseline - 1), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img_draw, detection_counts, violations


def convert_video_to_h264(input_path: str, output_path: str) -> bool:
    """Sử dụng imageio-ffmpeg để transcode video sang H.264 giúp phát mượt trong browser."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        return False

# ---------------------------------------------------------
# SIDEBAR: CẤU HÌNH MÔ HÌNH & THÔNG SỐ
# ---------------------------------------------------------
with st.sidebar:
    st.title(" Model Settings")
    
    # 1. Danh sách chọn Mô hình (Trung tính, không áp đặt)
    model_keys = list(AVAILABLE_MODELS.keys())
    selected_model_name = st.selectbox(
        " Model:",
        options=model_keys,
        index=0,
        help="Chọn giữa các mô hình đã huấn luyện: YOLO26, YOLO11 hoặc RT-DETR"
    )
    
    current_model_info = AVAILABLE_MODELS[selected_model_name]
    
    # Hiển thị thông số kỹ thuật của mô hình đang chọn
    st.markdown(f"""
    **Thông Tin Kỹ Thuật:**
    -  **Kiến trúc:** `{current_model_info['arch']}`
    -  **Định dạng:** `{current_model_info['type']}`
    -  **Dung lượng:** `{current_model_info['size']}`
    """)
    
    st.divider()
    
    # 2. Ngưỡng Confidence & IoU
    conf_thresh = st.slider("Confidence threshold:", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    iou_thresh = st.slider("IoU threshold:", min_value=0.2, max_value=0.8, value=0.45, step=0.05)
    
    st.divider()
    
    # 3. Lọc Class hiển thị
    all_class_names = list(CLASS_INFO.keys())
    selected_classes = st.multiselect(
        "Classes to display:",
        options=all_class_names,
        default=all_class_names,
        format_func=lambda x: f"{CLASS_INFO[x]['vn']} ({x})"
    )
    
    show_confidence_text = st.checkbox("Hiển thị điểm tin cậy trên nhãn", value=True)

# Tải mô hình đã chọn
with st.spinner(f"Đang tải {selected_model_name}..."):
    model = load_model(current_model_info["path"])

# ---------------------------------------------------------
# TIÊU ĐỀ CHÍNH
# ---------------------------------------------------------
st.markdown('<div class="main-title">PPE Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect personal protective equipment in images, videos and camera captures.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TABS GIAO DIỆN
# ---------------------------------------------------------
tab_image, tab_video, tab_webcam, tab_compare = st.tabs([
    "Image",
    "Video",
    "Camera",
    "Benchmark"
])

# =========================================================
# TAB 1: NHẬN DIỆN ẢNH
# =========================================================
with tab_image:
    st.subheader("Nhận Diện Trang Bị Bảo Hộ Qua Hình Ảnh")
    
    img_col1, img_col2 = st.columns([1, 2])
    
    with img_col1:
        img_source = st.radio("Chọn nguồn ảnh:", ["Tải ảnh lên từ máy tính", "Chọn ảnh mẫu có sẵn (Dataset)"], horizontal=True)
        
        input_image = None
        
        if img_source == "Tải ảnh lên từ máy tính":
            uploaded_file = st.file_uploader("Chọn ảnh cần phân tích (JPG, PNG, JPEG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            sample_dir = BASE_DIR / "data" / "raw" / "test" / "images"
            if sample_dir.is_dir():
                sample_files = list(sample_dir.glob("*.jpg"))[:20]
                if sample_files:
                    sample_dict = {f.name: f for f in sample_files}
                    chosen_sample_name = st.selectbox("Chọn ảnh mẫu từ tập test:", list(sample_dict.keys()))
                    sample_path = sample_dict[chosen_sample_name]
                    input_image = cv2.imread(str(sample_path))
                else:
                    st.warning("Chưa tìm thấy ảnh mẫu trong thư mục test.")
            else:
                st.warning("Không tìm thấy đường dẫn `data/raw/test/images`.")
                
        run_img_btn = st.button(" Phân Tích & Nhận Diện PPE", type="primary", use_container_width=True)

    with img_col2:
        if input_image is not None:
            if run_img_btn or "last_img_result" in st.session_state:
                with st.spinner("Đang chạy mô hình AI nhận diện..."):
                    t_start = time.time()
                    results = model.predict(
                        source=input_image,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False
                    )
                    inference_time = (time.time() - t_start) * 1000
                    
                    annotated_bgr, counts, violations = draw_ppe_boxes(
                        input_image, results, selected_classes=selected_classes, show_conf=show_confidence_text
                    )
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    st.session_state["last_img_result"] = (annotated_rgb, counts, violations, inference_time)
            
            if "last_img_result" in st.session_state:
                annotated_rgb, counts, violations, inf_time = st.session_state["last_img_result"]
                
                # Cảnh báo vi phạm
                total_violations = len(violations)
                if total_violations > 0:
                    st.markdown(f"""
                    <div class="alert-danger-box">
                         CẢNH BÁO VI PHẠM: Phát hiện {total_violations} lỗi vi phạm quy định an toàn (Không mũ bảo hộ / Không áo phản quang)!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-safe-box">
                         AN TOÀN LAO ĐỘNG: 100% công nhân được trang bị đầy đủ PPE hợp lệ!
                    </div>
                    """, unsafe_allow_html=True)
                
                # Hiển thị ảnh so sánh hoặc ảnh kết quả
                view_mode = st.radio("Chế độ hiển thị:", ["Ảnh Kết Quả Nhận Diện", "So Sánh Gốc vs Kết Quả"], horizontal=True)
                if view_mode == "Ảnh Kết Quả Nhận Diện":
                    st.image(annotated_rgb, caption=f"Mô hình: {selected_model_name} | Thời gian: {inf_time:.1f}ms", use_container_width=True)
                else:
                    c1, c2 = st.columns(2)
                    c1.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Ảnh Gốc", use_container_width=True)
                    c2.image(annotated_rgb, caption=f"Ảnh Nhận Diện ({inf_time:.1f}ms)", use_container_width=True)
                
                # Bảng số liệu thống kê nhanh
                st.markdown("###  Thống Kê Trang Bị Trong Ảnh")
                cols = st.columns(4)
                cols[0].metric("Công Nhân (Person)", counts.get("person", 0))
                cols[1].metric("Đội Mũ (Helmet)", counts.get("helmet", 0) + counts.get("helmet on", 0))
                cols[2].metric("Áo Phản Quang (Vest)", counts.get("vest", 0))
                cols[3].metric("Số Lỗi Vi Phạm ", total_violations, delta=-total_violations if total_violations > 0 else 0, delta_color="inverse")
                
                # Nút tải ảnh kết quả
                res_pil = Image.fromarray(annotated_rgb)
                temp_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                res_pil.save(temp_buf.name, format="JPEG", quality=95)
                with open(temp_buf.name, "rb") as f:
                    st.download_button(
                        label=" Tải Ảnh Kết Quả Về Máy",
                        data=f.read(),
                        file_name="ppe_detection_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        else:
            st.info(" Hãy tải lên một ảnh hoặc chọn ảnh mẫu từ cột bên trái để bắt đầu.")

# =========================================================
# TAB 2: NHẬN DIỆN VIDEO
# =========================================================
with tab_video:
    st.subheader("Nhận Diện Trang Bị Bảo Hộ Trên Video")
    
    vid_col1, vid_col2 = st.columns([1, 2])
    
    with vid_col1:
        uploaded_video = st.file_uploader("Tải lên tệp video (MP4, AVI, MOV, MKV):", type=["mp4", "avi", "mov", "mkv"])
        
        frame_skip = st.select_slider(
            "Tối ưu tốc độ xử lý (Frame Skip):",
            options=[1, 2, 3, 4, 5],
            value=2,
            help="Bỏ qua khung hình giúp xử lý video dài nhanh hơn nhiều lần mà vẫn đảm bảo độ chính xác theo dõi."
        )
        
        show_live_feed = st.checkbox("Hiển thị khung hình trực tiếp khi đang xử lý", value=True)
        start_video_btn = st.button(" Bắt Đầu Nhận Diện Video", type="primary", use_container_width=True)
        
    with vid_col2:
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
            tfile.write(uploaded_video.read())
            video_input_path = tfile.name
            
            st.video(video_input_path)
            
            if start_video_btn:
                cap = cv2.VideoCapture(video_input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                raw_out_path = tempfile.mktemp(suffix=".mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(raw_out_path, fourcc, fps / frame_skip, (width, height))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                live_preview = st.empty() if show_live_feed else None
                
                frame_idx = 0
                processed_count = 0
                total_violations_video = 0
                violation_history = []
                
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue
                        
                    results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
                    annotated_frame, counts, violations = draw_ppe_boxes(
                        frame, results, selected_classes=selected_classes, show_conf=show_confidence_text
                    )
                    
                    num_viol = len(violations)
                    if num_viol > 0:
                        total_violations_video += num_viol
                    violation_history.append(num_viol)
                    
                    header_color = (0, 0, 220) if num_viol > 0 else (0, 180, 0)
                    cv2.rectangle(annotated_frame, (10, 10), (340, 65), (20, 20, 20), -1)
                    cv2.rectangle(annotated_frame, (10, 10), (340, 65), header_color, 2)
                    status_str = f"VI PHAM: {num_viol} LOI" if num_viol > 0 else "PPE HOP LE - AN TOAN"
                    cv2.putText(annotated_frame, status_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2)
                    cv2.putText(annotated_frame, f"Model: {selected_model_name.split()[0]} | Frame: {frame_idx}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    
                    out_writer.write(annotated_frame)
                    processed_count += 1
                    
                    pct = min(1.0, frame_idx / max(1, total_frames))
                    progress_bar.progress(pct)
                    elapsed = time.time() - start_time
                    cur_fps = processed_count / max(0.001, elapsed)
                    status_text.markdown(f"⏳ **Đang xử lý:** Frame `{frame_idx}/{total_frames}` ({pct*100:.1f}%) — Tốc độ: `{cur_fps:.1f} FPS` — Vi phạm: `{total_violations_video}`")
                    
                    if show_live_feed and live_preview is not None and processed_count % 3 == 0:
                        live_preview.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                cap.release()
                out_writer.release()
                
                progress_bar.progress(1.0)
                status_text.success(f" Hoàn thành xử lý {processed_count} frames trong {time.time()-start_time:.1f}s!")
                
                final_h264_path = tempfile.mktemp(suffix=".mp4")
                with st.spinner("Đang chuẩn hóa định dạng video để hiển thị..."):
                    h264_success = convert_video_to_h264(raw_out_path, final_h264_path)
                    video_to_display = final_h264_path if h264_success else raw_out_path
                
                st.markdown("###  Video Kết Quả")
                st.video(video_to_display)
                
                with open(video_to_display, "rb") as f:
                    st.download_button(
                        label=" Tải Video Kết Quả Nhận Diện",
                        data=f.read(),
                        file_name=f"ppe_detected_{Path(uploaded_video.name).stem}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
        else:
            st.info(" Vui lòng tải lên một tệp video ở cột bên trái.")

# =========================================================
# TAB 3: CAMERA TRỰC TIẾP (WEBCAM)
# =========================================================
with tab_webcam:
    st.subheader("Kiểm Soát Trang Bị An Toàn Qua Camera")
    st.write("Sử dụng camera của thiết bị để kiểm tra nhanh công nhân trước khi vào công trường.")
    
    cam_col1, cam_col2 = st.columns(2)
    
    with cam_col1:
        camera_img = st.camera_input("Chụp ảnh kiểm tra an toàn:")
        
    with cam_col2:
        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner(f"Đang phân tích với {selected_model_name}..."):
                t0 = time.time()
                results = model.predict(cv_img, conf=conf_thresh, iou=iou_thresh, verbose=False)
                t_infer = (time.time() - t0) * 1000
                
                annotated_bgr, counts, violations = draw_ppe_boxes(
                    cv_img, results, selected_classes=selected_classes, show_conf=show_confidence_text
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                if len(violations) > 0:
                    st.error(f" CẢNH BÁO: Phát hiện {len(violations)} vi phạm! Yêu cầu trang bị đầy đủ trước khi vào khu vực thi công.")
                else:
                    st.success(" ĐỦ TIÊU CHUẨN AN TOÀN: Đạt chuẩn PPE!")
                    
                st.image(annotated_rgb, caption=f"Thời gian suy luận: {t_infer:.1f}ms", use_container_width=True)

# =========================================================
# TAB 4: SO SÁNH HIỆU NĂNG MÔ HÌNH (BENCHMARK)
# =========================================================
with tab_compare:
    st.subheader("So Sánh Hiệu Năng: YOLO26 Nano vs YOLO11 Small vs RT-DETR")
    st.markdown("""
    Bảng tổng hợp đối chiếu hiệu năng thực nghiệm giữa các kiến trúc trên cùng tập dữ liệu PPE:
    """)
    
    benchmark_data = {
        "Tiêu Chí So Sánh": [
            " Đánh Giá Tổng Quan",
            "Kiến Trúc Mô Hình (Backbone)",
            "Cơ Chế Hậu Xử Lý (Post-processing)",
            "Định Dạng Trọng Số",
            "Kích Thước Tệp (File Size)",
            "Độ Chính Xác mAP@0.50 (B)",
            "Độ Phủ (Recall - Bắt Lỗi Vi Phạm)",
            "Độ Chuẩn Xác (Precision - B)",
            "Tốc Độ Suy Luận (Inference)",
            "Ứng Dụng Tối Ưu Nhất"
        ],
        "YOLO26 Nano (yolo26n.pt)": [
            " TỐT NHẤT TOÀN DIỆN",
            "YOLO26 (CNN thế hệ mới)",
            "NMS (Non-Maximum Suppression)",
            "PyTorch (.pt)",
            "5.46 MB (Siêu nhẹ)",
            "72.20% (Cao nhất)",
            "80.67% (Bắt vi phạm tốt nhất)",
            "59.45%",
            "~5 ms / frame (Nhanh nhất)",
            "Video Real-time, Edge Device, CPU"
        ],
        "YOLO11 Small (yolo11.pt)": [
            " Xếp thứ hai",
            "YOLO11 (CNN Small)",
            "NMS (Non-Maximum Suppression)",
            "PyTorch (.pt)",
            "19.19 MB (~3.5x)",
            "70.25%",
            "80.45%",
            "60.92%",
            "~15 ms / frame",
            "Server GPU, Xử lý ảnh tĩnh"
        ],
        "RT-DETR (RTDETR.onnx)": [
            " Transformer Detector",
            "Vision Transformer (Deformable DETR)",
            "NMS-Free (Không cần NMS)",
            "ONNX Runtime (.onnx)",
            "131.1 MB (Lớn nhất)",
            "~71.50%",
            "~79.80%",
            "~62.10% (Chính xác cao)",
            "~30 ms / frame",
            "Phân tích ảnh phức tạp, Không lo trùng box"
        ]
    }
    df_bm = pd.DataFrame(benchmark_data)
    st.table(df_bm)
    
    st.markdown("""
    ###  Phân Tích & So Sánh Chuyên Sâu:
    1. **Tại sao `YOLO26 Nano` đạt kết quả tối ưu nhất cho bài toán giám sát an toàn?**
       - **mAP50 cao nhất (72.2%)**: Vượt trội hơn cả YOLO11 (70.25%) và RT-DETR (~71.5%).
       - **Độ phủ (Recall) đạt 80.67%**: Đây là chỉ số quan trọng nhất trong giám sát an toàn lao động, nhằm hạn chế tối đa việc bỏ sót công nhân không đội mũ bảo hộ hoặc không mặc áo phản quang.
       - **Kích thước chỉ 5.46 MB và tốc độ ~5ms**: Nhanh gấp 3-6 lần các mô hình khác, giúp chạy mượt mà ngay cả trên CPU thông thường và video fps cao.
    2. **Đặc điểm của `RT-DETR (ONNX)`:**
       - Sử dụng kiến trúc Vision Transformer tiên tiến, loại bỏ hoàn toàn thuật toán NMS truyền thống (NMS-Free), giúp triệt tiêu hiện tượng bounding box bị đè lặp khi đám đông công nhân đứng sát nhau.
       - Định dạng ONNX chuẩn mở cho phép triển khai tối ưu trên nhiều nền tảng phần cứng khác nhau.
    3. **Lựa chọn mô hình:**
       - Bạn có thể chuyển đổi linh hoạt mô hình bất kỳ lúc nào tại menu **Sidebar bên trái** để thử nghiệm và so sánh trực tiếp.
    """)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Đồ án Môn học: Nhận diện Trang Bị Bảo Hộ Lao Động (PPE Detection) &copy; 2026</p>", unsafe_allow_html=True)
