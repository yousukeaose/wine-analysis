import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# データ読み込み（パスはあなたの環境に合わせて変更）
df = pd.read_csv("../data/winequality-red.csv")

# 説明変数（自由に選んでOK）
X = df[["alcohol"]]   # ← 例：アルコール度数だけ使う
y = df["quality"]

# モデル作成
model = LinearRegression()
model.fit(X, y)

# 予測
y_pred = model.predict(X)

# 結果の図を作成
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression: Alcohol vs Quality")
plt.legend()

# 図を wine-analysis 直下に保存
plt.savefig("../linear_regression_result.png")
plt.close()

print("図を保存しました: linear_regression_result.png")
