# [1]
import pandas as pd
df=pd.read_csv("./day06/wine.csv")
data=df[["alcohol","sugar","pH"]] # 와인들의 속성 3개
target=df["class"] # 1: 화이트와인, 0: 레드와인 

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target=train_test_split(data,target,random_state=42)

# 트리의 앙상블: 학습한 모델에서 오답들을 서로 상쇄하고 정답을 강화하여 예측정확도 높여 과대적합 방지하는 방법 # 여러가지 방법 존재
# [2] 랜덤포레스트
# 결정트리는 전체 특성("alcohol", "sugar", "pH") 중에 가장 영향력 있는 특성으로 예측 결정하는 방법(한쪽 특성에만 과대적합)
# 랜덤포레스트 모든 특성 사용
    # 부트스트랩 샘플링: 전체 훈련 데이터 중에서 무작위로 샘플 선정
    # 무작위 특성: 전체 특성 중에서 무작위로 샘플 선정
# 즉) 모든 특성들을 사용하여 다양한 트리를 구성
# 
# oob(Out-of-Bad) 무작위(중복허용) 선정시 1번도 선정 안 된 자료들을 평가용으로 사용
# 예시[1][2][3][4][5] 중에서 무작위로 [1][3][5][5][2] 선정하면 1번도 선정 안 된 [4] 샘플
# [4] 샘플 가지고 학습 모델 검증 > oob_score, 자체 검증
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

# 교차 검증
from sklearn.model_selection import cross_validate
scores=cross_validate(rf, train_input, train_target, n_jobs=-1)
print(scores) # 'test_score': array([0.88      , 0.90051282, 0.90349076, 0.89014374, 0.88295688])}
import numpy as np
print(np.mean(scores["test_score"])) # 0.8914208392565683 # T4-01,T4-02 보다 점수 높다

# 특성 중요도
rf.fit(train_input,train_target)
print(rf.feature_importances_) # [0.23155241 0.49706658 0.27138101] # 즉) 결정트리보다 조금 더 골고루 분산됨
# 분류 모델중에서는 로지스틱 회귀 모델vs복잡한 모델은 트리 모델(앙상블)

# [3] 엑스트라 트리
# 랜덤포레스트 중복 허용한 무작위 샘플 선출
# 엑스트라 트리
    # 모든 트리가 전체 샘플 자료를 학습
    # 무작위 노드 분할: 예) sugar 특성을 무작위로 1:4 기준으로 분리한다 # 무작위라서 오답이 많이 발생한다
# 예시) "나이" 특성에 20세~60세가 존재한 경우 노드 분할 예시
#       Tree(노드)에서 무작위로 나이 특성을 29세 이상 조건을 만든다(수학적인 계산이 없어서 빠르다.)
# 즉) 노드마다 서로 다른 기준점을 분할하여 다양성 확보한다. 계산식이 없어서 허술한 방법이지만 학습수와 방대한 양으로 오차 극복

from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier(n_jobs=-1, random_state=42) # 모델 생성
scores=cross_validate(et, train_input, train_target, n_jobs=-1)
print(scores) # array([0.89128205, 0.89128205, 0.89938398, 0.88706366, 0.88295688])}
print(np.mean(scores["test_score"])) # 0.8903937240035804
# 특성 주우요도
et.fit(train_input, train_target)
print(et.feature_importances_) # [0.20702369 0.51313261 0.2798437 ]

# [4] 그레이디언트 부스팅
# 랜덤포레스트: 중복 허용한 무작위 샘플/특성 선정하여 학습
# 엑스트라트리: 무작위로 노드 분할 기준 선정 학습
# 그레이디언트 부스팅: 부모노드(트리)가 예측하고 오차를 자식노드(트리)에게 넘겨 학습
# 자식 노드가 많아질수록 오차는 줄어든다. (과대적합 주의)

# 예시) Tree(1노드)에서 실제 정답이 10을 목표로 하여 예측한 결과가 7이면 오차 3발생
#  Tree(2노드)에서 실제 정답이 10을 목표로 하여 예측한 결과가 7이면 오차 3발생
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=42) # 모델 객체 생성
scores=cross_validate(gb, train_input, train_target, n_jobs=-1)
print(scores) # ([0.86461538, 0.87794872, 0.88090349, 0.8613963 , 0.87268994])}
print(np.mean(scores["test_score"])) # 0.8715107671247301

# 특성 중요도
gb.fit(train_input, train_target)
print(gb.feature_importances_) # [0.12517641 0.73300095 0.14182264]
# dt(결정트리)/rf(랜덤포레스트)/et(엑스트라트리)보다 뾰족하게 한쪽 특성에 집중된 결과