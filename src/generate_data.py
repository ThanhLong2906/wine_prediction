import pandas as pd
import numpy as np
import os
def generate_data(n_samples=1000, path="data/prices.csv", random_state=42):
    if not os.path.exists(path) or os.path.getsize(path) > 0:

        np.random.seed(random_state)

        # feature giả lập
        size = np.random.randint(5, 20, n_samples)           # kích thước (cm)
        color = np.random.randint(1, 4, n_samples)           # màu: 1=đỏ, 2=hồng, 3=trắng
        freshness = np.random.randint(1, 11, n_samples)      # độ tươi (1-10)

        # công thức tính giá giả lập
        price = 5 * size + 10 * freshness + color * 5 + np.random.normal(0, 10, n_samples)

        df = pd.DataFrame({
            "size": size,
            "color": color,
            "freshness": freshness,
            "price": price.round(2)
        })

        df.to_csv(path, index=False)
        print(f"✅ Đã sinh {n_samples} dòng dữ liệu và lưu vào {path}")


if __name__ == "__main__":
    generate_data()
