# T1-01.py
# 캐글의 데이터셋: https://www.kaggle.com/datasets/vipullrathod/fish-market?select=Fish.csv

# [1] csv 불러오기
import pandas as pd
df=pd.read_csv("./day01/Fish.csv")
print(df.head())

# [2] 특정한 물고기 추출
bream_df=df[df["Species"]=="Bream"]

# [3] 특정한 물고기 추출, 도미/Bream
bream_length=bream_df["Length2"].tolist() # 도미 길이 추출
bream_weight=bream_df["Weight"].tolist() # 도미 무게 추출
print(bream_length, bream_weight)

# [4] 특정한 물고기 추출, 빙어/Smelt
smelt_df=df[df["Species"]=="Smelt"]
smelt_length=smelt_df["Length2"].tolist()
smelt_weight=smelt_df["Weight"].tolist()

# [5] 시각화
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight) # 도미 시각화
plt.scatter(smelt_length, smelt_weight) # 빙어 시각화
plt.xlabel("length(cm)")
plt.ylabel("weight(gram)")
plt.show()