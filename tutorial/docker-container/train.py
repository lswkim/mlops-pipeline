import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 데이터셋 로드
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(data_url, names=columns)
print(data.head())
# 데이터 전처리
X = data.drop('class', axis=1)
y = data['class']

# 훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 모델 평가
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

while True:
	time.sleep(3)
