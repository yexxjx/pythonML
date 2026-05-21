# [1] 여러가지 특성에 분류 모델
import pandas as pd
df=pd.read_csv("./day05/Fish.csv")

# 어종 7개, Species
fish_target=df["Species"]

# 특성 6개, Weight,Length1,Length2,Length3,Height,Width
fish_input=df[["Weight","Length1","Length2","Length3","Height","Width"]]

# 훈련 / 테스트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target=train_test_split(fish_input, fish_target, test_size=0.25, random_state=42)

# 스케일링
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

# [*] 경사 하강법
# fit() 모델 학습에서는 정답(target)도 같이 학습 중. 예측(y)값과 실제 정답 간의 오차(기울기/가중치 조절하면서 오차 줄이기) 측정
# 예) 산꼭대기에서 내려가는 방법 중에 가장 최적의 경로로 내려오는 방법=경사 하강법(수많은 경우의 수 계산하여 판단, 컴퓨터가 좋아야한다)
# (1) 경사 하강법(정확도 좋지만 학습 속도가 느리다) vs (2) 확률 경사 하강법(SGD: 정확도 낮지만 학습 속도가 빠름: 미니배치)

# [*] 로그 로스/손실 함수, 손실(예측과 정답의 전체 차이)
# 로그 로스 함수는 0과 1의 확률 값이 아닌 *오차 값*을 측정

# [*] 에포크
# 학습 횟수

# [2] SGDClassifier, 분류 모델
from sklearn.linear_model import SGDClassifier

# loss="log_loss", 로스 함수
# random_state: SGD가 전체 데이터 학습이 아닌 일부 자료(미니배치) 가지고 학습하는데 사용되는 분리 기준(난수값)
# max_iter: (반복)계산 횟수 # 미니 배치이므로 전체 데이터 셋을 "10" 이면 10번 반복 학습하여 모델 정확도 향상 / 최적의 정확도에서 멈춤 (에포크)
# tol=None: 최적의 정확도를 찾아도 계속 반복 학습 설정
sc=SGDClassifier(loss="log_loss", random_state=42, max_iter=10,tol=None) # 모델 객체 생성
sc.fit(train_scaled, train_target) # 모델 학습
print(sc.score(test_scaled, test_target)) # 0.875
print(sc.predict(test_scaled[:3])) # ['Perch' 'Smelt' 'Pike']

# [3] 점진적 학습(중간 학습 가능하다)
sc.partial_fit(train_scaled, train_target) # (위에서 이미 학습된 모델에) 10번+1번 > 11번 학습
print(sc.score(test_scaled, test_target))

# [4] 최적의 학습횟수(에포크) 찾기
sc=SGDClassifier(loss="log_loss", random_state=42) # max_iter 생략시 1학습

train_score=[] # 학습용 정확도
test_score=[] # 테스트용 정확도

#
import numpy as np
classes=np.unique(train_target) # 정답자의 중복제거한 고유 정답만 추출

for i in range(0,150): # 150번 반복
    sc.partial_fit(train_scaled, train_target, classes=classes) # 1학습
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# 정확도 시각화 # 과대적합 # 과소적합 # 최적의 에포크(반복횟수) 는 학습용과 테스트용이 고르게 오르는 시점
import matplotlib.pyplot as plt
plt.plot(train_score) # 
plt.plot(test_score) # 
plt.show() 