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
    initial_sidebar_state="collapsed",
)

# Giao diện tối giản
st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 650; margin-bottom: 0.2rem; color: inherit; }
    .sub-title { color: #7a7a7a; font-size: 0.95rem; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 14px; font-weight: 500; }
    div[data-testid="stMetric"] { padding: 4px 0; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
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

AVAILABLE_MODELS = {}
if YOLO26N_PATH:
    AVAILABLE_MODELS["YOLO26 Nano (yolo26n.pt - Tốt nhất & Khuyên dùng)"] = {
        "path": YOLO26N_PATH,
        "mAP50": "72.2%",
        "recall": "80.7%",
        "precision": "59.4%",
        "size": "5.46 MB",
        "speed": "Rất nhanh (~5ms/ảnh)",
        "tag": "Khuyên dùng (mAP50 cao nhất 72.2%, siêu nhẹ 5.4MB)",
    }
if YOLO11_PATH:
    AVAILABLE_MODELS["YOLO11 Small (yolo11.pt)"] = {
        "path": YOLO11_PATH,
        "mAP50": "70.2%",
        "recall": "80.4%",
        "precision": "60.9%",
        "size": "19.19 MB",
        "speed": "Trung bình (~15ms/ảnh)",
        "tag": "Mô hình so sánh (19.2MB)",
    }

if not AVAILABLE_MODELS:
    st.error(" Không tìm thấy file trọng số `yolo26n.pt` hoặc `yolo11.pt` trong thư mục `models/` hoặc `notebooks/`!")
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
    st.title("Cấu hình")
    model_keys = list(AVAILABLE_MODELS.keys())
    selected_model_name = st.selectbox("Mô hình", options=model_keys, index=0)
    current_model_info = AVAILABLE_MODELS[selected_model_name]
    st.caption(f"mAP50: {current_model_info['mAP50']} · Recall: {current_model_info['recall']} · Size: {current_model_info['size']}")

    with st.expander("Thông số nâng cao", expanded=False):
        conf_thresh = st.slider("Confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        iou_thresh = st.slider("IoU (NMS)", min_value=0.2, max_value=0.8, value=0.45, step=0.05)
        all_class_names = list(CLASS_INFO.keys())
        selected_classes = st.multiselect(
            "Lớp cần hiển thị", options=all_class_names, default=all_class_names,
            format_func=lambda x: f"{CLASS_INFO[x]['vn']} ({x})"
        )
        show_confidence_text = st.checkbox("Hiện confidence trên nhãn", value=True)

 # Tải mô hình đã chọn
with st.spinner("Đang khởi tạo mô hình..."):
    model = load_model(current_model_info["path"])

# ---------------------------------------------------------
# TIÊU ĐỀ CHÍNH
# ---------------------------------------------------------
st.markdown('<div class="main-title">PPE Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Nhận diện trang bị bảo hộ lao động trên ảnh, video và camera</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TABS GIAO DIỆN
# ---------------------------------------------------------
tab_image, tab_video, tab_webcam, tab_compare = st.tabs([
    "Ảnh",
    "Video",
    "Camera",
    "Benchmark"
])

# =========================================================
# TAB 1: NHẬN DIỆN ẢNH
# =========================================================
with tab_image:
    st.subheader("Phân tích ảnh")
    
    img_col1, img_col2 = st.columns([1, 2])
    
    with img_col1:
        img_source = st.radio("Nguồn ảnh:", ["Tải ảnh lên", "Ảnh mẫu"], horizontal=True)
        
        input_image = None
        
        if img_source == "Tải ảnh lên":
            uploaded_file = st.file_uploader("Tải ảnh (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
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
                
        run_img_btn = st.button("Phân tích ảnh", type="primary", use_container_width=True)

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
                    st.error(f"Phát hiện {total_violations} trường hợp vi phạm PPE.")
                else:
                    st.success("Không phát hiện vi phạm PPE.")
                
                # Hiển thị ảnh so sánh hoặc ảnh kết quả
                view_mode = st.radio("Hiển thị:", ["Kết quả", "So sánh ảnh gốc và kết quả"], horizontal=True)
                if view_mode == "Kết quả":
                    st.image(annotated_rgb, caption=f"Kết quả nhận diện ({inf_time:.1f}ms)", use_container_width=True)
                else:
                    c1, c2 = st.columns(2)
                    c1.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Ảnh Gốc", use_container_width=True)
                    c2.image(annotated_rgb, caption=f"Ảnh Nhận Diện ({inf_time:.1f}ms)", use_container_width=True)
                
                # Bảng số liệu thống kê nhanh
                st.markdown("### Thống kê")
                cols = st.columns(4)
                cols[0].metric("Công Nhân (Person)", counts.get("person", 0))
                cols[1].metric("Đội Mũ (Helmet)", counts.get("helmet", 0) + counts.get("helmet on", 0))
                cols[2].metric("Áo Phản Quang (Vest)", counts.get("vest", 0))
                cols[3].metric("Số Lỗi Vi Phạm ️", total_violations, delta=-total_violations if total_violations > 0 else 0, delta_color="inverse")
                
                # Nút tải ảnh kết quả
                res_pil = Image.fromarray(annotated_rgb)
                temp_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                res_pil.save(temp_buf.name, format="JPEG", quality=95)
                with open(temp_buf.name, "rb") as f:
                    st.download_button(
                        label="Tải ảnh kết quả",
                        data=f.read(),
                        file_name="ppe_detection_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        else:
            st.info("Hãy tải lên một ảnh hoặc chọn ảnh mẫu từ cột bên trái để bắt đầu.")

# =========================================================
# TAB 2: NHẬN DIỆN VIDEO
# =========================================================
with tab_video:
    st.subheader("Phân tích video")
    
    vid_col1, vid_col2 = st.columns([1, 2])
    
    with vid_col1:
        uploaded_video = st.file_uploader("Tải video (MP4, AVI, MOV, MKV):", type=["mp4", "avi", "mov", "mkv"])
        
        frame_skip = st.select_slider(
            "Frame skip:",
            options=[1, 2, 3, 4, 5],
            value=2,
            help="Bỏ qua khung hình giúp xử lý video dài nhanh hơn nhiều lần mà vẫn đảm bảo độ chính xác theo dõi."
        )
        
        show_live_feed = st.checkbox("Hiện preview khi xử lý", value=True)
        start_video_btn = st.button(" Xử lý video", type="primary", use_container_width=True)
        
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
                    cv2.rectangle(annotated_frame, (10, 10), (320, 65), (20, 20, 20), -1)
                    cv2.rectangle(annotated_frame, (10, 10), (320, 65), header_color, 2)
                    status_str = f"VI PHAM: {num_viol} LOI" if num_viol > 0 else "PPE HOP LE - AN TOAN"
                    cv2.putText(annotated_frame, status_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2)
                    cv2.putText(annotated_frame, f"Model: {Path(current_model_info['path']).stem} | Frame: {frame_idx}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    
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
                
                st.markdown("###  Kết quả video")
                st.video(video_to_display)
                
                with open(video_to_display, "rb") as f:
                    st.download_button(
                        label="Tải video kết quả",
                        data=f.read(),
                        file_name=f"ppe_detected_{Path(uploaded_video.name).stem}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
        else:
            st.info("Vui lòng tải lên một tệp video ở cột bên trái.")

# =========================================================
# TAB 3: CAMERA TRỰC TIẾP (WEBCAM)
# =========================================================
with tab_webcam:
    st.subheader("Camera")
    st.write("Sử dụng camera của thiết bị để kiểm tra nhanh công nhân trước khi vào công trường.")
    
    cam_col1, cam_col2 = st.columns(2)
    
    with cam_col1:
        camera_img = st.camera_input("Chụp ảnh:")
        
    with cam_col2:
        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("Đang kiểm tra PPE..."):
                t0 = time.time()
                results = model.predict(cv_img, conf=conf_thresh, iou=iou_thresh, verbose=False)
                t_infer = (time.time() - t0) * 1000
                
                annotated_bgr, counts, violations = draw_ppe_boxes(
                    cv_img, results, selected_classes=selected_classes, show_conf=show_confidence_text
                )
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                if len(violations) > 0:
                    st.error(f" Phát hiện {len(violations)} vi phạm PPE.")
                else:
                    st.success(" Không phát hiện vi phạm PPE.")
                    
                st.image(annotated_rgb, caption=f"Thời gian suy luận: {t_infer:.1f}ms", use_container_width=True)

# =========================================================
# TAB 4: SO SÁNH HIỆU NĂNG MÔ HÌNH (BENCHMARK)
# =========================================================
with tab_compare:
    st.subheader("So sánh mô hình: YOLO26 Nano và YOLO11 Small")
    st.markdown("""
    Dựa trên kết quả huấn luyện thực nghiệm 100 epochs trên cùng bộ dữ liệu PPE (`data.yaml`):
    """)
    
    benchmark_data = {
        "Tiêu Chí So Sánh": [
            " Đánh Giá Tổng Quan",
            "Kiến Trúc Mô Hình (Backbone)",
            "Kích Thước Trọng Số (.pt)",
            "Độ Chính Xác mAP@0.50 (B)",
            "Độ Chính Xác Toàn Diện mAP@0.50-0.95",
            "Độ Phủ (Recall - Bắt Lỗi Vi Phạm)",
            "Độ Chuẩn Xác (Precision - B)",
            "Tốc Độ Suy Luận (Inference Speed)",
            "Khả Năng Triển Khai Video/Webcam"
        ],
        "YOLO26 Nano (yolo26n.pt)": [
            " TỐT NHẤT (Khuyên dùng)",
            "YOLO26 Nano (Thế hệ mới nhất)",
            "5.46 MB (Siêu nhẹ)",
            "72.20% (Cao hơn)",
            "44.20%",
            "80.67% (Phát hiện vi phạm tốt hơn)",
            "59.45%",
            "~5 ms / frame (Nhanh gấp ~3x)",
            "Rất mượt trên mọi thiết bị & CPU"
        ],
        "YOLO11 Small (yolo11.pt)": [
            " Xếp sau",
            "YOLO11 Small",
            "19.19 MB (Nặng hơn 3.5x)",
            "70.25%",
            "44.20%",
            "80.45%",
            "60.92%",
            "~15 ms / frame",
            "Đòi hỏi tài nguyên phần cứng hơn"
        ]
    }
    df_bm = pd.DataFrame(benchmark_data)
    st.table(df_bm)
    
    st.markdown("""
    ###  Kết luận & Khuyến nghị thực nghiệm:
    1. **Tại sao nên chọn `yolo26n.pt`?**
       - **mAP50 cao hơn (+1.95%)**: Đạt **72.2%** so với 70.25% của YOLO11.
       - **Recall cao hơn**: Đạt **80.67%**, đây là chỉ số tối quan trọng trong an toàn lao động nhằm **tránh bỏ sót các trường hợp công nhân không đội mũ hoặc không mặc áo bảo hộ**.
       - **Dung lượng siêu nhẹ (5.4MB)**: Nhẹ hơn 3.5 lần, giúp khởi động ứng dụng nhanh, tốn ít RAM và đạt FPS cao khi chạy video thời gian thực.
    2. **Chuyển đổi linh hoạt**: Bạn có thể chuyển đổi qua lại giữa hai mô hình bất kỳ lúc nào tại thanh cài đặt bên trái (**Sidebar**).
    """)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Đồ án Môn học: Nhận diện Trang Bị Bảo Hộ Lao Động (PPE Detection) &copy; 2026</p>", unsafe_allow_html=True)
