"""
power_bi_script.py
-------------------------------------------------------------------
Cách dùng trong Power BI:
1. Trong Power Query Editor, chọn query chứa dữ liệu khách hàng
   (vd query đọc từ sample_customers.csv).
2. Vào menu Transform > Run Python script.
3. Power BI sẽ tự tạo sẵn 1 biến DataFrame tên 'dataset' chứa đúng
   dữ liệu của query đó — bạn không cần tự khai báo biến này.
4. Dán toàn bộ nội dung file này vào ô script rồi bấm OK.
5. Sau khi chạy xong, Power BI đọc lại biến 'dataset' và hiển thị
   nó như 1 bảng kết quả mới — bảng này sẽ có thêm 2 cột
   risk_probability và risk_level.
-------------------------------------------------------------------
"""

import pandas as pd
import joblib

# 1. Load "bundle" model đã train sẵn từ file .pkl.
#    Bundle là 1 dict chứa model + scaler + danh sách cột, được lưu lại
#    lúc train ở file model2_loan_default_prediction.py.
#    Lưu ý: sửa đường dẫn bên dưới cho đúng vị trí file trên máy bạn,
#    vì thư mục làm việc của Power BI khi chạy script không cố định.
bundle = joblib.load(r'C:\Users\Dell\Desktop\loan-default-prediction\loan_default_model.pkl')

model = bundle['model']                       # model đã train (vd RandomForest, XGBoost...)
scaler = bundle['scaler']                     # bộ chuẩn hoá dữ liệu số, đã fit lúc train
feature_columns = bundle['feature_columns']   # danh sách cột model cần, đúng thứ tự lúc train

# 2. Copy dataset ra 1 bản làm việc riêng (df) để không đụng vào 'dataset' gốc.
#    Ta sẽ dùng 'dataset' gốc ở bước cuối để gắn kết quả, giữ nguyên tên cột
#    thân thiện mà bạn đang thấy trên Power BI.
df = dataset.copy()

# 3. Đổi tên cột từ tên "thân thiện" (age, annual_income...) sang tên gốc
#    mà model đã được train (person_age, person_income...).
#    Đây là bước bắt buộc vì model chỉ nhận đúng tên cột lúc train.
rename_map = {
    'age': 'person_age',
    'gender': 'person_gender',
    'annual_income': 'person_income',
    'education': 'person_education',
    'years_employment': 'person_emp_exp',
    'home_ownership': 'person_home_ownership',
    'credit_score': 'credit_score',
    'loan_intent': 'loan_intent',
    'credit_history_length': 'cb_person_cred_hist_length',
    'previous_defaults': 'previous_loan_defaults_on_file',
}
df = df.rename(columns=rename_map)

# 4. Tạo thêm 2 cột đặc trưng (feature engineering) — phải tính giống hệt
#    lúc train model, nếu không dự đoán sẽ sai lệch.
#    Income_per_Emp_Year: thu nhập trung bình theo mỗi năm kinh nghiệm làm việc.
#    +1 ở mẫu số để tránh chia cho 0 khi years_employment = 0.
df['Income_per_Emp_Year'] = df['person_income'] / (df['person_emp_exp'] + 1)

#    Credit_History_per_Age: tỉ lệ số năm có lịch sử tín dụng so với tuổi.
#    +1 ở mẫu số để tránh chia cho 0.
df['Credit_History_per_Age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)

# 5. One-hot encode các cột dạng chữ (categorical) thành các cột 0/1.
#    Model chỉ hiểu số, nên các giá trị như "male"/"female", "RENT"/"OWN"...
#    phải được tách thành các cột riêng (vd person_gender_male = 1 hoặc 0).
cat_cols = ['person_gender', 'person_education', 'person_home_ownership',
            'loan_intent', 'previous_loan_defaults_on_file']
encoded = pd.get_dummies(df, columns=cat_cols)

# 6. Căn chỉnh lại đúng bộ cột & đúng thứ tự mà model yêu cầu (feature_columns).
#    Nếu dữ liệu mới thiếu 1 giá trị hiếm (vd không có ai chọn "OTHER" trong
#    home_ownership) thì cột tương ứng sẽ được tự động điền giá trị 0.
encoded = encoded.reindex(columns=feature_columns, fill_value=0)

# 7. Chuẩn hoá dữ liệu bằng scaler đã lưu sẵn từ lúc train (chỉ transform,
#    KHÔNG fit lại, để giữ đúng thang đo model đã học).
scaled = scaler.transform(encoded)

# 8. Dự đoán xác suất rủi ro vỡ nợ cho từng dòng.
#    predict_proba trả về 2 cột: [xác suất KHÔNG vỡ nợ, xác suất VỠ NỢ].
#    Ta chỉ lấy cột thứ 2 (index 1) — xác suất vỡ nợ.
probabilities = model.predict_proba(scaled)[:, 1]

# 9. Gắn kết quả vào 'dataset' gốc (không phải 'df' đã đổi tên cột), để bảng
#    hiển thị trên Power BI vẫn giữ nguyên tên cột quen thuộc.
#    Đổi xác suất (0.0–1.0) sang phần trăm (0–100) và làm tròn 2 chữ số.
dataset['risk_probability'] = (probabilities * 100).round(2)

# 10. Phân loại mức rủi ro theo ngưỡng (threshold) 50%:
#     >= 50% -> "Cao", < 50% -> "Thấp".
dataset['risk_level'] = dataset['risk_probability'].apply(
    lambda p: 'Cao' if p >= 50 else 'Thấp'
)

# Xong! Power BI sẽ tự động đọc lại biến 'dataset' sau khi script chạy xong
# và hiển thị nó như 1 bảng mới, kèm 2 cột risk_probability và risk_level.
