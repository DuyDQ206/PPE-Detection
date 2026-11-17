# YOLO11s PPE checkpoint

Model nhận diện 4 lớp theo đúng ID: `0 boots`, `1 helmet`, `2 person`, `3 vest`.

`best.pt` được đặt tại đây sau khi giải nén ZIP Kaggle, được Git bỏ qua. File `metadata.json`
lưu SHA256, kích thước checkpoint, nguồn ZIP và phiên bản thư viện để kiểm tra khi chia sẻ trọng số.

Khôi phục từ ZIP: giải nén file kết quả Kaggle, rồi chép
`ppe_yolo11s/weights/best.pt` vào `models/yolo11s/best.pt`.

Hoặc tải `best.pt` từ asset của GitHub Release do chủ repo xuất bản, đặt vào thư mục này
và kiểm tra SHA256:

```powershell
Get-FileHash models/yolo11s/best.pt -Algorithm SHA256
```

Hash phải khớp `sha256` trong `metadata.json`. Chỉ cần checkpoint và thư viện runtime
để predict; dataset chỉ cần khi train hoặc evaluate.
