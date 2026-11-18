# YOLO11s PPE checkpoint

Checkpoint mới từ lần chạy `ppe_yolo11s-2`, huấn luyện 100 epochs trên dataset PPE-5.

Model nhận diện 8 lớp theo đúng ID:

| ID | Lớp |
| ---: | --- |
| 0 | boots |
| 1 | helmet |
| 2 | helmet on |
| 3 | no boots |
| 4 | no helmet |
| 5 | no vest |
| 6 | person |
| 7 | vest |

`best.pt` được đặt tại đây sau khi giải nén ZIP Kaggle, được Git bỏ qua. File `metadata.json`
lưu SHA256, kích thước checkpoint, nguồn ZIP và phiên bản thư viện để kiểm tra khi chia sẻ trọng số.

Khôi phục từ ZIP: giải nén `ppe_yolo11s-2_results.zip`, rồi chép
`ppe_yolo11s-2/weights/best.pt` vào `models/yolo11s/best.pt`.

Kiểm tra SHA256:

```powershell
Get-FileHash models/yolo11s/best.pt -Algorithm SHA256
```

Hash phải khớp `sha256` trong `metadata.json`. Chỉ cần checkpoint và thư viện runtime
để predict; dataset chỉ cần khi train hoặc evaluate.
