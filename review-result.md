# Kết quả Review: AI24-ML-M3.1-GA-2T-TĐ

> Tài liệu: Giáo án Học máy (Machine Learning) — Module M2.1 — 6 bài học, 2073 dòng.
> Ngày review: 2026-04-24.
> Phương pháp: Review có hệ thống (phân khúc → review theo lớp → review chuyên sâu theo chủ đề).

---

## Phân khúc tài liệu

| Khúc | Bài học | Dòng | Chủ đề chính |
|------|---------|------|--------------|
| 1 | Máy học phân loại | 5–384 | ML, Supervised/Unsupervised, Training data, Prediction, Loss, Epoch |
| 2 | Mô hình AI kiểm thử CAPTCHA | 385–726 | CAPTCHA, Turing Test, Image Recognition, Pattern Recognition |
| 3 | Camera AI giao thông | 727–1186 | Image Classification, Labeling, Computer Vision, Confidence, Threshold |
| 4 | Chatbot phân tích phản hồi | 1187–1469 | Text Classification, Sentiment Analysis |
| 5 | Máy học âm thanh | 1470–1713 | Pre-trained model, Sound Classification |
| 6 | Máy bắn muỗi thông minh | 1714–2073 | Regression, Predict number, Linear/Multiple/Decision Tree Regression |

---

## Tổng hợp vấn đề phát hiện


### Vấn đề 1 — Dữ liệu hồi quy tuyến tính hoàn hảo (Khúc 6, phần 2.2.1)

**Vị trí:** Bài "Máy bắn muỗi thông minh", phần 2.2.1 — Tìm hiểu về bài toán dự đoán.

**Nội dung gốc:**
> Bảng dữ liệu: 20 phút → 6 điểm, 30 phút → 7 điểm, 40 phút → 8 điểm, 50 phút → 9 điểm.
> HS trả lời: "Dữ liệu có hoàn toàn có nằm trên 1 đường thẳng."
> GV dẫn dắt: "Trong thực tế, dữ liệu với số lượng lớn sẽ không hoàn hảo..."

**Vấn đề:** Dữ liệu mẫu có quan hệ tuyến tính hoàn hảo (y = 0.1x + 4, R² = 1.0). Điều này **không bao giờ xảy ra trong thực tế** và tạo ấn tượng sai lệch cho học sinh rằng dữ liệu thực luôn "sạch" và tuân theo quy luật chính xác. Mặc dù GV có nhắc "trong thực tế dữ liệu sẽ không hoàn hảo", nhưng bản thân dữ liệu mẫu lại mâu thuẫn với phát biểu này.

**Mức độ:** Gây nhầm (misleading).

**Đề xuất sửa:** Thêm nhiễu vào dữ liệu mẫu, ví dụ: 20 phút → 5.5, 30 phút → 7.2, 40 phút → 7.8, 50 phút → 9.1. Như vậy dữ liệu vẫn có xu hướng tăng rõ ràng nhưng không nằm hoàn hảo trên đường thẳng, phù hợp hơn với thực tế và nhất quán với lời giải thích của GV.

---

### Vấn đề 2 — Ví dụ "dự báo thời tiết" dùng cho Regression có thể gây nhầm (Khúc 6, phần 2.2.1)

**Vị trí:** Bài "Máy bắn muỗi thông minh", phần 2.2.1.

**Nội dung gốc:**
> "Dự báo thời tiết: Dựa vào dữ liệu các ngày trước để dự đoán khả năng mưa, giúp chúng ta chuẩn bị áo mưa hoặc sắp xếp kế hoạch phù hợp."

**Vấn đề:** Ví dụ này được đặt trong ngữ cảnh giới thiệu bài toán dự đoán (dẫn đến Regression). Tuy nhiên, "dự đoán khả năng mưa" (mưa hay không mưa) thực chất là bài toán **phân loại (Classification)**, không phải hồi quy. Ngay sau đó, GV cũng phân biệt: "Dự đoán ngày mai có mưa hay không?" là classification. Điều này tạo mâu thuẫn nội bộ trong cùng một phần.

**Mức độ:** Gây nhầm (misleading).

**Đề xuất sửa:** Thay ví dụ "dự báo thời tiết" bằng ví dụ rõ ràng hơn về dự đoán giá trị số, ví dụ: "Dự đoán nhiệt độ ngày mai: Dựa vào dữ liệu nhiệt độ các ngày trước để ước lượng nhiệt độ ngày mai sẽ là bao nhiêu độ C." Hoặc nếu giữ ví dụ thời tiết, cần nói rõ: "dự đoán nhiệt độ" thay vì "dự đoán khả năng mưa".

---

### Vấn đề 3 — Thuật ngữ "hồi quy tuyến tính bội" vs "hồi quy đa biến" (Khúc 6, phần 2.2.2)

**Vị trí:** Bài "Máy bắn muỗi thông minh", phần 2.2.1.2 và phần Ôn tập Tiết 2.

**Nội dung gốc:**
> Phần 2.2.1.2: "Hồi quy tuyến tính bội" (Multiple Linear Regression).
> Phần Ôn tập Tiết 2: "Hồi quy tuyến tính, **đa biến**, cây quyết định".

**Vấn đề:** Thuật ngữ không nhất quán. Phần dạy kiến thức dùng "hồi quy tuyến tính bội" nhưng phần ôn tập lại dùng "đa biến". Trong tiếng Việt chuyên ngành, "Multiple Linear Regression" thường được dịch là "hồi quy tuyến tính bội" (nhiều biến đầu vào). "Hồi quy đa biến" (Multivariate Regression) thực ra chỉ bài toán có **nhiều biến đầu ra** — đây là hai khái niệm khác nhau.

**Mức độ:** Thiếu chính xác (imprecise) + Mâu thuẫn (contradictory).

**Đề xuất sửa:** Thống nhất dùng "hồi quy tuyến tính bội" xuyên suốt. Sửa phần Ôn tập Tiết 2 thành: "Hồi quy tuyến tính, **hồi quy tuyến tính bội**, cây quyết định".

---

### Vấn đề 4 — Lỗi chính tả "Traning loss" (Khúc 1, phần 2.2)

**Vị trí:** Bài "Máy học phân loại", phần 2.2 — Giới thiệu biểu đồ Loss vs Epochs.

**Nội dung gốc:**
> "Nếu đường **Traning** loss đi xuống thì mô hình đang học..."
> "Nếu đường **Traning** loss nằm ngang..."

**Vấn đề:** Viết sai chính tả "Traning" thay vì "Training". Lỗi này xuất hiện 2 lần liên tiếp, trong khi dòng tiếp theo lại viết đúng "Training loss".

**Mức độ:** Thiếu chính xác (imprecise).

**Đề xuất sửa:** Sửa "Traning" thành "Training" ở cả 2 vị trí.

---

### Vấn đề 5 — Bảng CAPTCHA: Thử thách 9 và 10 có cùng đáp án (Khúc 2, phần 1.2.1)

**Vị trí:** Bài "Mô hình AI kiểm thử CAPTCHA", phần 1.2.1 — Bảng thử thách CAPTCHA.

**Nội dung gốc:**
> | 9 | [image10] | oxgjYyp |
> | 10 | [image11] | oxgjYyp |

**Vấn đề:** Thử thách 9 và 10 có cùng đáp án "oxgjYyp". Nếu đây là 2 hình ảnh CAPTCHA khác nhau thì đáp án phải khác nhau. Nếu cùng hình ảnh thì không cần lặp lại. Rất có thể đây là lỗi copy-paste — đáp án thử thách 10 bị sao chép nhầm từ thử thách 9.

**Mức độ:** Sai (factually wrong).

**Đề xuất sửa:** Kiểm tra lại hình ảnh CAPTCHA thứ 10 và cập nhật đáp án chính xác.

---

### Vấn đề 6 — Quy trình ML for Kids: Đáp án câu trắc nghiệm cần xác minh (Khúc 1, phần 5.2)

**Vị trí:** Bài "Máy học phân loại", phần 5.2 — Câu hỏi trắc nghiệm.

**Nội dung gốc:**
> **Câu hỏi:** Quy trình huấn luyện mô hình học máy trên ML for kids gồm những bước nào?
> 1. Make → Train → Learn & Test
> 2. **Train → Learn & Test → Make**
> 3. Learn & Test → Train → Make
> 4. Train → Make → Learn & Test

**Vấn đề:** Đáp án đúng được đánh dấu là phương án 2: "Train → Learn & Test → Make". Tuy nhiên, trên nền tảng Machine Learning for Kids, quy trình thực tế là: (1) **Train** (thêm dữ liệu huấn luyện vào các nhãn), (2) **Learn & Test** (huấn luyện mô hình và kiểm tra), (3) **Make** (sử dụng mô hình trong Scratch). Đáp án này đúng với giao diện ML for Kids. Tuy nhiên, cần lưu ý rằng bước "Train" ở đây có nghĩa là "chuẩn bị dữ liệu huấn luyện" chứ không phải "huấn luyện mô hình" — điều này có thể gây nhầm lẫn với thuật ngữ "Training model" đã dạy ở phần quy trình ML.

**Mức độ:** Gây nhầm (misleading).

**Đề xuất sửa:** Thêm ghi chú giải thích: "Lưu ý: Bước 'Train' trên ML for Kids có nghĩa là chuẩn bị/thêm dữ liệu huấn luyện, bước 'Learn & Test' mới là bước huấn luyện mô hình và kiểm tra."

---

### Vấn đề 7 — Nhầm lẫn giữa "nhóm cảm xúc" và "nhóm chủ đề sản phẩm" (Khúc 4, phần 1.2.2)

**Vị trí:** Bài "Chatbot phân tích phản hồi từ khách hàng", phần 1.2.2 — Giao nhiệm vụ.

**Nội dung gốc:**
> "Nhận diện được tin nhắn với phân loại rõ ràng thành các nhóm **cảm xúc** như kiểu dáng, màu sắc, chất liệu, độ bền,..của sản phẩm."

**Vấn đề:** "Kiểu dáng, màu sắc, chất liệu, độ bền" không phải là **nhóm cảm xúc** — đây là **nhóm chủ đề/khía cạnh** (aspect) của sản phẩm. Cảm xúc (sentiment) là tích cực/tiêu cực/trung lập. Bài toán thực tế ở đây là **Aspect-based Sentiment Analysis** (phân tích cảm xúc theo khía cạnh), nhưng cách diễn đạt hiện tại nhầm lẫn giữa "cảm xúc" và "chủ đề".

**Mức độ:** Sai (factually wrong).

**Đề xuất sửa:** Sửa thành: "Nhận diện được tin nhắn với phân loại rõ ràng thành các **nhóm chủ đề** (kiểu dáng, màu sắc, chất liệu, độ bền,...) kèm **cảm xúc** (tích cực/tiêu cực) về sản phẩm." Hoặc đơn giản hơn: "...phân loại thành các nhóm phản hồi tiêu cực, tích cực về kiểu dáng, màu sắc, chất liệu, độ bền,..." (như đã viết đúng ở phần II. Sản phẩm).

---

### Vấn đề 8 — Thiếu sót: Không giải thích "Hồi quy" nghĩa là gì (Khúc 6, phần 2.2.1)

**Vị trí:** Bài "Máy bắn muỗi thông minh", phần 2.2.1.

**Nội dung gốc:**
> "Dự đoán số (predict number) trong Machine Learning thường được gọi là bài toán Hồi quy (Regression)..."

**Vấn đề:** Thuật ngữ "Hồi quy" được giới thiệu nhưng không giải thích tại sao lại gọi là "hồi quy" (regression — nghĩa gốc là "quay về/trở lại"). Đối với học sinh, từ "hồi quy" rất trừu tượng và không gợi liên tưởng đến "dự đoán số". Một câu giải thích ngắn về nguồn gốc thuật ngữ sẽ giúp học sinh nhớ lâu hơn.

**Mức độ:** Thiếu sót (missing).

**Đề xuất sửa:** Thêm ghi chú: "Thuật ngữ 'Regression' (hồi quy) có nguồn gốc từ nghiên cứu thống kê của Francis Galton vào thế kỷ 19, khi ông phát hiện rằng chiều cao của con cái có xu hướng 'quay về' (regress) gần mức trung bình so với cha mẹ. Ngày nay, thuật ngữ này được dùng chung cho các bài toán dự đoán giá trị số liên tục."

---

### Vấn đề 9 — Mô tả Confidence chưa hoàn toàn chính xác (Khúc 3, phần 2.2.3)

**Vị trí:** Bài "Camera AI giao thông", phần 2.2.3.

**Nội dung gốc:**
> "Độ tin cậy (Confidence): là giá trị xác suất (thường từ 0 đến 1 hoặc 0% đến 100%) mà mô hình gán cho một dự đoán, thể hiện mức độ chắc chắn của mô hình về kết quả đó."

**Vấn đề:** Gọi confidence là "giá trị xác suất" không hoàn toàn chính xác về mặt kỹ thuật. Confidence score của nhiều mô hình ML (đặc biệt sau softmax) **trông giống** xác suất (nằm trong [0,1], tổng = 1) nhưng **không phải xác suất thực sự** theo nghĩa thống kê (calibrated probability). Tuy nhiên, đối với đối tượng học sinh phổ thông, mức đơn giản hóa này có thể chấp nhận được.

**Mức độ:** Thiếu chính xác (imprecise) — nhưng chấp nhận được cho đối tượng học sinh.

**Đề xuất sửa:** Không bắt buộc sửa. Nếu muốn chính xác hơn, có thể đổi thành: "...là một con số (thường từ 0% đến 100%) mà mô hình đưa ra cùng với dự đoán, thể hiện mức độ 'tự tin' của mô hình về kết quả đó."

---

### Vấn đề 10 — Sai số heading: "3.2" thay vì "2.2" (Khúc 6, Tiết 2)

**Vị trí:** Bài "Máy bắn muỗi thông minh", Tiết 2, phần 2.

**Nội dung gốc:**
> "### 2\. Chế tạo mẫu, thử nghiệm và đánh giá (55')"
> "#### 2.1. Mục tiêu"
> "#### **3.2. Tổ chức thực hiện**"

**Vấn đề:** Heading "3.2. Tổ chức thực hiện" nằm trong mục 2, nên phải là "2.2. Tổ chức thực hiện". Đây là lỗi đánh số heading.

**Mức độ:** Thiếu chính xác (imprecise).

**Đề xuất sửa:** Sửa "3.2" thành "2.2".

---

### Vấn đề 11 — Bài "Máy học phân loại" ghi Module M2.1 nhưng tiêu đề chung ghi M3.1 (Toàn tài liệu)

**Vị trí:** Tên file: "AI24-ML-**M3.1**-GA-2T-TĐ.md". Bài 1: "Module: **M2.1**".

**Nội dung gốc:**
> Tên file chứa "M3.1" nhưng tất cả các bài học bên trong đều ghi "Module: M2.1" hoặc "Module: Làm quen với học máy (Machine learning)".

**Vấn đề:** Mã module trong tên file (M3.1) không khớp với mã module trong nội dung (M2.1). Cần xác minh mã module chính xác.

**Mức độ:** Mâu thuẫn (contradictory).

**Đề xuất sửa:** Thống nhất mã module giữa tên file và nội dung.

---

### Vấn đề 12 — Bài 1 ghi "Học phần: Học máy" nhưng các bài sau ghi "Học phần: Làm quen với học máy" (Toàn tài liệu)

**Vị trí:** Bài 1 + 2: "Học phần: Học máy (Machine learning)". Bài 3, 4, 5, 6: "Module: Làm quen với học máy (Machine learning)".

**Vấn đề:** Tên học phần/module không nhất quán giữa các bài. Bài 1 và 2 ghi "Học phần: Học máy" còn bài 3–6 ghi "Module: Làm quen với học máy". Ngoài ra, bài 1–2 dùng từ "Học phần" còn bài 3–6 dùng từ "Module" — hai cách gọi khác nhau cho cùng một thông tin.

**Mức độ:** Mâu thuẫn (contradictory).

**Đề xuất sửa:** Thống nhất tên và cách gọi xuyên suốt tất cả các bài.

---

### Vấn đề 13 — Dữ liệu cây quyết định quá ít và quá "sạch" (Khúc 6, phần 2.2.1.3)

**Vị trí:** Bài "Máy bắn muỗi thông minh", phần 2.2.1.3 — Thử thách xây dựng cây quyết định.

**Nội dung gốc:**
> Bảng 6 dòng dữ liệu, mỗi tổ hợp (Thời gian, Giờ ngủ, Làm bài tập) cho ra đúng 1 giá trị điểm duy nhất.

**Vấn đề:** Dữ liệu chỉ có 6 mẫu và hoàn toàn không có nhiễu — mỗi tổ hợp đầu vào cho ra đúng 1 đầu ra. Trong thực tế, cây quyết định hồi quy hoạt động bằng cách **lấy trung bình** các giá trị trong mỗi nhánh lá, nhưng ở đây mỗi nhánh lá chỉ có 1 mẫu nên không thể hiện được cơ chế "trung bình". Tuy nhiên, với đối tượng học sinh phổ thông và thời lượng hạn chế, mức đơn giản hóa này có thể chấp nhận được.

**Mức độ:** Gây nhầm (misleading) — nhưng chấp nhận được cho mục đích sư phạm.

**Đề xuất sửa:** Không bắt buộc sửa. Nếu muốn cải thiện, có thể thêm 2–3 mẫu dữ liệu nữa để mỗi nhánh lá có ít nhất 2 mẫu, giúp minh họa cơ chế "lấy trung bình".

---

## Bảng tóm tắt

| # | Vị trí | Vấn đề tóm tắt | Mức độ |
|---|--------|----------------|--------|
| 1 | Khúc 6, 2.2.1 | Dữ liệu hồi quy tuyến tính hoàn hảo R²=1.0 | Gây nhầm |
| 2 | Khúc 6, 2.2.1 | Ví dụ "dự báo khả năng mưa" dùng cho Regression nhưng thực chất là Classification | Gây nhầm |
| 3 | Khúc 6, 2.2.1.2 + Ôn tập | "Hồi quy tuyến tính bội" vs "đa biến" — thuật ngữ không nhất quán và sai nghĩa | Thiếu chính xác + Mâu thuẫn |
| 4 | Khúc 1, 2.2 | Lỗi chính tả "Traning" (2 lần) | Thiếu chính xác |
| 5 | Khúc 2, 1.2.1 | Thử thách CAPTCHA 9 và 10 cùng đáp án "oxgjYyp" | Sai |
| 6 | Khúc 1, 5.2 | Bước "Train" trên ML for Kids dễ nhầm với "Training model" | Gây nhầm |
| 7 | Khúc 4, 1.2.2 | Nhầm "nhóm cảm xúc" với "nhóm chủ đề sản phẩm" | Sai |
| 8 | Khúc 6, 2.2.1 | Không giải thích nguồn gốc thuật ngữ "Hồi quy" | Thiếu sót |
| 9 | Khúc 3, 2.2.3 | Gọi Confidence là "giá trị xác suất" — chưa hoàn toàn chính xác | Thiếu chính xác |
| 10 | Khúc 6, Tiết 2 | Heading "3.2" thay vì "2.2" | Thiếu chính xác |
| 11 | Toàn tài liệu | Mã module M3.1 (tên file) vs M2.1 (nội dung) | Mâu thuẫn |
| 12 | Toàn tài liệu | Tên học phần không nhất quán giữa các bài | Mâu thuẫn |
| 13 | Khúc 6, 2.2.1.3 | Dữ liệu cây quyết định quá ít, không có nhiễu | Gây nhầm |

---

**Tổng cộng: 13 vấn đề** — 2 sai, 4 gây nhầm, 3 thiếu chính xác, 1 thiếu sót, 3 mâu thuẫn.

**Đánh giá tổng thể:** Tài liệu có chất lượng tốt về mặt sư phạm, cấu trúc rõ ràng, hoạt động phong phú. Các vấn đề phát hiện chủ yếu ở mức "gây nhầm" và "thiếu chính xác" — không có lỗi kiến thức nghiêm trọng nào làm sai lệch hoàn toàn nội dung. Hai lỗi "sai" (vấn đề 5 và 7) cần được sửa ưu tiên.
