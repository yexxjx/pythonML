# [1] 숭어 길이, 높이, 두께(3가지특성) 무게(타겟)
import pandas as pd
df=pd.read_csv("./day03/Fish.csv")
perch=df[df["Species"]=="Perch"]
perch_full=perch[["Length2", "Height", "Width"]]
perch_weight=perch["Weight"].values

# [2] 훈련 세트와 테스트 세트 분리 *모델 검증 용도*
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target=train_test_split(perch_full, perch_weight, test_size=0.2, random_state=42)

# [3] 특성 공학, 다항 특성 제공
# 다양한 특성을 추가로 만들어서 모델이 다양한 구조로 이해하기 위한 방법 (여러 개 재료/자료 만들기)
from sklearn.preprocessing import PolynomialFeatures

# 예시1) 1(기본값), 2, 2*2
poly=PolynomialFeatures() # 객체 생성
poly.fit([[2]])
print(poly.transform([[2]])) # [[1. 2. 4.]]

# 예시2) 1(기본값 제외) 2, 3, 2*2, 3*3
poly=PolynomialFeatures(include_bias=False) # 기본 편항 없음
poly.fit([[2,3]])
print(poly.transform([[2,3]])) # [[2. 3. 4. 6. 9.]]

# 적용: 길이 높이 두께: 3가지의 특성을 가짐
poly=PolynomialFeatures(include_bias=False) # 다항 틍성 객체 생성
poly.fit(train_input) # 학습할 속성들을 대입
train_poly=poly.transform(train_input)
test_poly=poly.transform(test_input)
print(train_poly.shape) # 3가지 특성 > 9가지 특성으로 변환
# T2-02(직접 제곱**). T2-03(PolynomialFeatures)

# [4] 다항 회귀 모델 학습
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(train_poly, train_target) # 학습

# [5] 평가
print(lr.score(test_poly, test_target)) # 0.9764933250721679 테스트 세트는 훈련 이후에 평가 목적으로 사용

# ====================================== 
# 스코어(점수), 회귀 모델에서는 결정 계수
# 계수란? 기울기와 가중치 의미 # 즉) 어떠한 예측 결과에 얼마나 중요한 비중 차지하는지
# 결정계수란? 0~1 사이의 값으로 예측한 값이 얼마나 잘 설명하는지 나타내는 수치
# 결정계수 계산식 *왜? K-NN모델은 전체 계산식이 아닌 근접한 이웃 이용한 계산식 이므로
    # 타깃의 총 변동량=SS_TOT=sum((실제값-실제값평균)**2)
    # 타깃의 오차 변동량=SS_RES((실제값-예측값)**2)
    # 1(100%)-(ss_res/ss_tot)

# [6] 과대적합 확인 (5차항으로 표현) , degree=차수
poly=PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly=poly.transform(train_input)
test_poly=poly.transform(test_input)
print(train_poly.shape) # 2차항에서는 3>9 # 5차항에서는 3>55

# 모델 학습
lr.fit(train_poly,train_target) # 55항으로 학습

# 모델 평가
# 과대적합: 특정한 자료에만 과도한 학습을 통해 학습된 것만 예측하고 새로운 자료에 대해서는 에측 불가능하다
# 적절한 차수 선택한 모델의 최적화 조절한다.
print(lr.score(train_poly, train_target)) # 0.999999999999193
print(lr.score(test_poly, test_target)) # -74.97494194987092

# [7] 규제하기 위한 전처리(스케일링)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() # 스케일링 객체 생성

# 과대 적합된 자료들을 스케일링 # 서로 다른 특성들 간에 (의미)단위가 다르므로 동일한 단위 사용하기 위해 (0~1)
ss.fit(train_poly) # 3개 특성이 55개 특성으로 과대 적합된 상태의 표준편차/평균 계산
train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)

# 릿지/라쏘 회귀들은 과적합된 자료들을 자동으로 제거 해준다
# [8] 릿지 회귀: 가중치 줄여가면서 완전한 선 만들기 목적
# 알파(alpha=규제단위), alpha 단위가 크면 클수록 가중치(기울기) # 0으로 가깝게 만든다. # 규제 강도
# 예) 길이 50 > 10 줄었을 때 성능이 줄면 중요한 계수이다. 
# 예) 너비 50 > 10 줄었을 때 성능이 차이가 없으면 중요한 계수가 아니다. 

from sklearn.linear_model import Ridge
ridge=Ridge() # 릿지 모델 객체 생성
ridge.fit(train_scaled, train_target) # 스케일링된 과대적합 자료 학습

alpha_list=[0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge=Ridge(alpha=alpha) # 0.001~100까지 반복하면서 알파값 대입
    ridge.fit(train_scaled, train_target) # 서로 다른 알파값에 따른 학습
    print("======================================", alpha)
    print(ridge.score(train_scaled, train_target)) # 평가1
    print(ridge.score(test_scaled, test_target)) # 평가2
    # 최적: 학습 평가와 테스트 평가 간의 격차가 적은 것 =alpha10 이 가장 적합한 하이퍼파라미터임

# [9] 라쏘 회귀: 서로 특성 간의 관계없는 특성들을 0 제거
# 알파(alpha=규제강도)
# 특정한 특성의 값을 변경했을 때 결정계수의 교차가 거의 없으면 관계 없다.
# 즉) 관계없는 특성은 0으로 제거한다.
# 예) 길이 50 > 30 줄였을 때 성능/오차 없다 <필요없는 특성> 0으로 변경
# 예) 너비 50 > 30 줄였을 때 성능/오차 있다 <중요한 특성> 그대로

from sklearn.linear_model import Lasso
lasso=Lasso(alpha=10) # 라쏘 모델 객체 생성
lasso.fit(train_scaled, train_target) # 라쏘 모델 학습

print("======================================", 10)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))