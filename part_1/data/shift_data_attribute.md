Shift

Kỹ thuật

Mục tiêu giả lập môi trường thực tế

Shift 1 — Camera Shift

White balance, color temperature, slight exposure change, slight saturation/contrast change, mild sensor noise, mild sharpening/blur

Ảnh từ camera khác: khác hãng/cảm biến/cấu hình camera, khiến màu sắc, exposure và sharpness của ảnh thay đổi nhưng object vẫn giữ nguyên

Shift 2 — Lighting & Environment Shift

Brightness, gamma, contrast, shadow, mild haze, saturation, color temperature

Điều kiện ánh sáng và môi trường khác: ngày/đêm, trong nhà/ngoài trời, ánh sáng yếu hoặc ánh sáng không đồng đều

Shift 3 — Image Acquisition / Quality Shift

Downsampling → upsampling, JPEG compression, defocus blur, motion blur, mild Gaussian noise

Chất lượng ảnh khi deploy: camera độ phân giải thấp hơn, nén ảnh qua mạng, motion/focus khác hoặc sensor noise trong hệ thống production

Shift 4 — Appearance / Color Shift

Hue shift, saturation shift, channel-wise color scaling, contrast, gamma, slight brightness shift

Domain có appearance khác: white balance khác, màu sắc môi trường khác, camera processing pipeline khác hoặc dữ liệu được thu thập trong domain hình ảnh khác

Shift 5 — Mixed Realistic Shift

Kết hợp nhẹ Camera Shift + Lighting Shift + Image Quality Shift + Color Shift; mỗi transformation ở severity vừa phải

Production environment tổng hợp: dữ liệu thực tế đồng thời chịu nhiều thay đổi nhỏ như camera khác + ánh sáng khác + compression + noise, tạo covariate shift nhưng vẫn giữ semantic/class