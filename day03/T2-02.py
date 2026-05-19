import pandas as pd
df=pd.read_csv("./day03/Fish.csv")
fish_data=df[df["Species"].isin(["Perch"])]

# [1] 숭어의 길이, 무게 
perch_length=fish_data["Length2"].values
perch_weight=fish_data["Weight"].values
print(perch_length, perch_weight)

# [2] 훈련 세트와 테스트 세트 분리
from sklearn.model_selection import train_test_split
# 학습용, 테스트용, 학습용타깃, 테스트용타깃=train_test_split(특성, 타깃, test_size=비율, random_state=분리기준난수)
train_input, test_input, train_target, test_target=train_test_split(perch_length, perch_weight, test_size=0.2, random_state=42)

# [3] 학습 하기 전에 사이킷런 모델들은 2차원 배열만 가능하다 # [1,2,3] > [[1],[2],[3]]
train_input=train_input.reshape(-1,1) # reshape(행개수, 열개수) # -1행은 자동으로 설정, 1열은 1개 설정
test_input=test_input.reshape(-1,1)

# [4] k 최근접 이웃 회귀 모델 확인
from sklearn.neighbors import KNeighborsRegressor #회귀
knr=KNeighborsRegressor() # 모델 객체 생성
knr.fit(train_input, train_target) # 모델 학습
print(knr.score(test_input, test_target)) # 모델 평가

# [5] 임의의 값으로 예측하기, 임의의 물고기 길이 50 넣어서 무게 예측
print(knr.predict([[50]])) # [1010.]
print(knr.predict([[100]])) # [1010.]

# 문제점: k 최근접 이웃의 문제점은 단순한 주변 이웃의 평균으로 예측하기 때문에 최댓값을 벗어나면 항상 동일한 값으로 예측한다
# 즉) 소규모 또는 간단한 예측 프로그램에서만 사용된다

# [1] 다른 모델 사용하기
from sklearn.linear_model import LinearRegression # 선형회귀 모델
lr=LinearRegression() # 모델 객체 생성
lr.fit(train_input, train_target) # 모델 학습 
print(lr.score(test_input, test_target)) # 모델 평가
print(lr.predict([[50]])) # [1238.3175398]
print(lr.predict([[100]])) # [3191.00026354]

# 직선 공식(1차 방정식):y=ax+b
# 즉) (물고기)무게=가중치*(물고기)길이+절편
print(lr.coef_) # [39.05365447] # 직선의 기울기 (특성의 가중치)
    # 기울기(가중치 공식): x와 y의 편차 곱의 합/x의 편차 제 곱합
print(lr.intercept_) # -714.3651839448922 # 편항 # x(물고기 길이)가 0 일 때 y의 값
    # y절편 공식: y평균-(기울기*x의 평균)
# x와 y 가 직선 관계 이며 실 자료들은 물고기가 길이 1씩 증가할 때 무게가 꼭 비례 증가 하지 않는다. <애매하다>
# 즉) 초반에는 길이에 따라 무게가 3배 증가 하다가 중/후 반에는 무게가 2/1배 증가 할 수 있다. # 사람 키(어릴때 키가 자라고 나이들면 어느정도 고정)

# [2] 시각화
import matplotlib.pyplot as plt
plt.scatter(train_input, train_target) # x길이 y무게
plt.scatter(50,1238) # 예측된 결과 # 길이가 50일때 무게는 1238일 것이다
plt.scatter(100,3191)
plt.plot([15,100], lr.predict([[15],[100]])) # 회귀선 그리기
plt.show()

print(lr.score(test_input, test_target))

# [3] (다항: 여러 개 행) 선형 회귀 모델 # 2차 방정식
# 직선 공식(1차 방정식): y(예측)=W(가중치)x(특성)+B(절편)
# 직선 공식(2차 방정식): y(예측)=(W(가중치)x X(특성)제곱) + (W(가중치) X(특성))+B(절편)
# x(특성) 제곱: 물고기 길이에 제곱 *최적의 제곱수 찾아서 정확도 최적화 한다.
# x(특성): 물고기 길이
# 가중치: 기울기
# 절편: y절편/편향
# 즉) x제곱 항목이 추가되면서 그래프가 U 또는 곡선 모양으로 나온다. 길이가 커질 수록 무게는 뻥튀기되는 효과

import numpy as np
train_poly=np.column_stack((train_input**2, train_input)) # [길이제곱, 길이]
print(train_poly) # [[ 784.     28.  ] [ 745.29   27.3 ] [ 384.16   19.6 ] [ 484.     22.  ]]

# 모델 생성
lr=LinearRegression()
lr.fit(train_poly, train_target) # 다항으로 학습

# 예측할 자료, 길이: 50인 무게 예측
print(lr.predict([[50**2, 50]])) # [1579.0440311]

# 여러 개 예측
point=np.arange(15,50) # 15부터 50까지 1씩 증가하는 리스트 반환
point_poly=np.column_stack((point**2, point)) # 15~50 제곱한 열, 15~50 열
print(point_poly)

#  시각화
plt.scatter(train_input, train_target)
plt.plot(point,lr.predict(point_poly))
plt.show()

test_poly=np.column_stack((test_input**2, test_input))
print(lr.score(test_poly, test_target)) # 다항 회귀 평가 # 0.9801885585527479