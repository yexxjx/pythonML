from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd 
# 서비스  
class Service : 
    def __init__(self):
        self.model = None
    # 1. 
    def 학습요청( self , carList ) :
        # 1) df 만들기 
        df = pd.DataFrame( carList )
        train_data = df[['평균연비','누적주행거리키로','출고후경과월수','사고감가건수','소유자변경횟수']]
        target_data =df['매매가격만원'].values
        train_input , test_input , train_target , test_target = train_test_split( train_data , target_data , test_size=0.2, random_state=42)
        # 2) 학습모델 만들기 
        lr = LinearRegression()
        lr.fit( train_input , train_target )
        # 3) 학습모델 결정계수확인
        print( lr.score( test_input , test_target ) ) # 0.8466731821471493
        # 4) 모델 저장
        self.model = lr 
        return True
    # 2.
    def 예측요청( self , car ) :
        # 1) 만약에 모델이 없으면 
        if self.model is None : 
            return "학습 모델이 없습니다."
        # 2) 만약에 모델이 있으면 입력받은 딕셔너리 리스트로 변경
        del car['차량번호ID']
        del car['매매가격만원']
        car = [ value for value in car.values() ]
        # 3) 입력받은 리스트로 모델 예측
        predict = self.model.predict( [car]  )
        # 4) 예측된 값 반환 
        return predict[0]
service = Service()