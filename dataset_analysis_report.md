# Báo cáo Phân tích Tổng quan Dữ liệu (Dataset Analysis Report)

Báo cáo này tóm tắt kết quả phân tích cấu trúc, thống kê và các vấn đề bất thường trong tập dữ liệu thô **[train.csv](file:///C:/Users/NDHoang/PycharmProjects/ML-_Project_Autism_Prediction/data/raw/train.csv)** (3743 dòng, 22 cột) và **[test.csv](file:///C:/Users/NDHoang/PycharmProjects/ML-_Project_Autism_Prediction/data/raw/test.csv)** (200 dòng, 21 cột).

---

## 1. Thông tin cấu trúc & Giá trị khuyết thiếu (Missing Values)
* **Kích thước tập Train:** 3,743 dòng, 22 cột.
* **Kích thước tập Test:** 200 dòng, 21 cột.
* **Giá trị khuyết thiếu:** **Không có** bất kỳ giá trị NaN/khuyết thiếu nào trong cả hai tập dữ liệu thô (0%).

---

## 2. Phân tích các đặc trưng số (Numerical Features)
Tập dữ liệu chứa 14 cột dạng số (bao gồm ID, 10 câu hỏi sàng lọc `A1_Score` -> `A10_Score`, tuổi `age`, tổng điểm sàng lọc `result` và nhãn mục tiêu `Class/ASD`).

### Thống kê mô tả đặc trưng `age` và `result`
* **`result` (Điểm sàng lọc tổng hợp):**
  * Giá trị dao động từ `0` đến `10` (trung bình `3.92`, trung vị `4.0`).
  * Thực chất đây là tổng điểm của 10 cột `A1_Score` đến `A10_Score`.
* **`age` (Tuổi):**
  * Trung bình: `11.25` tuổi. Trung vị: `8.0` tuổi.
  * **Điểm bất thường (Outliers):** Giá trị lớn nhất (`max`) của trường tuổi lên đến **383.0 tuổi**! Đây rõ ràng là lỗi nhập liệu thô (data entry error) và cần phải được xử lý (ví dụ: gán bằng giá trị trung vị hoặc loại bỏ).

---

## 3. Phân tích các đặc trưng phân loại (Categorical Features)

### A. Các cột không có giá trị thông tin (Zero Variance)
> [!WARNING]
> Hai cột sau đây có giá trị giống nhau hoàn toàn trên 100% dữ liệu tập train, không mang lại bất kỳ giá trị thông tin nào cho mô hình:
> * **`contry_of_res` (Quốc gia):** 100% dữ liệu có giá trị là **`Unknown`**.
> * **`used_app_before` (Từng dùng app trước đó):** 100% dữ liệu có giá trị là **`no`**.
> 
> *Lưu ý về rò rỉ dữ liệu:* Việc áp dụng Target Encoding cho cột `contry_of_res` trong file EDA cũ thực chất đã biến một cột không có thông tin thành một cột rò rỉ nhãn (Target Leakage) do sự sai khác nhỏ về giá trị trung bình mục tiêu giữa các fold.

### B. Trùng lặp phân nhóm do lỗi chính tả/định dạng (`ethnicity`)
Cột sắc tộc chứa các phân nhóm trùng lặp do viết hoa/viết thường và dấu gạch ngang:
* **`White European`** (23.59%) và **`White-European`** (6.22%) -> Cần gộp lại.
* **`Others`** (2.67%) và **`others`** (0.03%) -> Cần gộp lại.

### C. Phân phối các cột khác
* **`gender` (Giới tính):** Nam (`m`) chiếm đa số với **68.10%** (2,549 ca), Nữ (`f`) chiếm **31.90%** (1,194 ca).
* **`jaundice` (Vàng da sơ sinh):** Tỷ lệ phân bổ rất cân bằng với **50.57%** (1,893 ca) đã từng bị vàng da sơ sinh.
* **`austim` (Tiền sử gia đình mắc tự kỷ):** **24.47%** (916 ca) có người thân trong gia đình bị tự kỷ.
* **`age_desc` (Mô tả nhóm tuổi):**
  * `Children` (Trẻ em): **59.47%** (2,226 ca)
  * `Adolescent` (Thanh thiếu niên): **19.24%** (720 ca)
  * `Adult` (Người lớn): **11.09%** (415 ca)
  * `Young` (Người trẻ): **10.21%** (382 ca)
* **`relation` (Người điền đơn sàng lọc):**
  * `Relative` (Họ hàng): **49.61%**
  * `Health care professional` (Nhân viên y tế): **33.82%**
  * `Self` (Tự điền): **14.16%**
  * `Parent` (Cha mẹ): **1.34%**
  * `Others` (Khác): **1.07%**

---

## 4. Phân phối nhãn mục tiêu (`Class/ASD`)
Nhãn phân loại của tập dữ liệu huấn luyện rất cân bằng, không có hiện tượng mất cân bằng nghiêm trọng:
* **Nhãn `1` (Mắc ASD):** 1,991 ca (**53.19%**)
* **Nhãn `0` (Không mắc ASD):** 1,752 ca (**46.81%**)

> [!NOTE]
> Do phân phối nhãn rất cân bằng (53% vs 47%), các kỹ thuật sinh dữ liệu ảo nhằm giải quyết mất cân bằng như SMOTE hay CTGAN là **không thực sự cần thiết** cho tập dữ liệu thô này và có thể làm giảm độ tổng quát hóa của mô hình (như kết quả giảm ROC-AUC mà chúng ta đã thấy trước đó).
