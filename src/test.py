import joblib
import pandas as pd

# load model
model = joblib.load("model/model.pkl")

# test với 1 sample
# nhập dữ liệu
a = input("kích thước: ")
b = input("Màu: ")
c = input("Độ tươi: ")
a = int(a)
b = int(b)
c = int(c)

sample = pd.DataFrame([[a, b, c]], columns=["size", "color", "freshness"])
pred = model.predict(sample)
print("Giá dự đoán:", pred)
