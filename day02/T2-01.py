# [1] Fish.csv 가져오기
import pandas as pd
df=pd.read_csv("./day02/Fish.csv")

# [2] Perch(농어)만 추출
target_fish=df[df["Species"].isin(["Perch"])]
target_fish.info() # 56마리

# 농어의 길이/무게 추출
perch_length=target_fish["Length2"].values
perch_weight=target_fish["Weight"].values
print(perch_length, perch_weight)

# 농어 길이에 따른 무게 예측
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.show()

# [3] 학습 모델 만들기, (1)준비: 학습용과 테스트용 분리한다. 왜? 모델평가에 사용된다
from sklearn.model_selection import train_test_split
# train_test_split(학습자료, 정답자료, test_size=분리비율, random_state=분리난수)
# random_state=분리할 때 사용되는 난수값,  난수값에 따라 분리한다, 고정값 넣어주면 항상 동일한 분리값 얻을 수 있음, 0~32 사이
train_input, test_input, train_target, test_target=train_test_split(perch_length, perch_weight, test_size=0.3, random_state=42)

# (2) 자료 형식(모양) 구성 (사이킷런) 모델 학습, 대부분 2차원 사용한다.
import numpy as np
array=np.array([1,2,3,4])
print(array.shape) # shape 배열의 모양 반환(행, 열) # (4,)
array2=np.array([[1,2],[3,4],[5,6]])
print(array2.shape) # (3, 2)
print(train_input.shape) # (39,) 1차원 배열 > 사이킷런 모델들은 1차원 배열 학습이 불가능하다

# T1-01(zip 활용), T1-02(column_stack 활용), T2-01(reshape): 1차원 > 2차원
# .reshape(행개수, 열개수) : 행개수에는 -1 넣어서 자동 뜻(자료개수만자동), 열 개수는 1개
train_input=train_input.reshape(-1,1)
print(train_input)  #[ [22.5] [19. ] [37. ] [22. ] [25.6] [42. ] [34.5]]]
print(train_input.shape) # (39, 1) 2차원 배열
# train_target=train_target.reshape(-1,1)
test_input=test_input.reshape(-1,1) # 테스트 학습 2차원

# [5] 모델 학습
from sklearn.neighbors import KNeighborsClassifier # K최근접이웃 모델 찾기
from sklearn.neighbors import KNeighborsRegressor # K최근접이웃 회귀

knr=KNeighborsRegressor() # 모델 객체 생성
knr.fit(train_input,train_target) # 모델 학습 # (길이, 무게) # 길이에 따른 무게 학습
print(knr.score(test_input, test_target)) # 모델 평가 # 0.9929281790592219 # 회귀모델에서는 결정 개수라고 한다

print(test_input)  # 길이 [ 8.4] [18. ] [27.5] [21.3] [22.5]
print(knr.predict(test_input)) # 무게 예측 [  61.4   78.   248.   117.   139.   847.   311.4  183.4  847.   112. ]
