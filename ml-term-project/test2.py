import pandas as pd

# DATA PREPROCESSING=======================================#

file_path = 'updated_steam_data.csv'

data = pd.read_csv(file_path)


#The Codes below are about Converting String values into Integer values.==#
#Purpose: To use 'Name of the steam game' value as GAMEID.

# getting the unique value of 'Name of the steam game' 
unique_games = data['Name of the steam game'].unique()

# convert 'Name of the steam game' into integer id value
game_id_map = {game: idx for idx, game in enumerate(unique_games)}

#The converted 'Name of the steam game' value is stored in Game ID column (Integer type)  
data['Game ID'] = data['Name of the steam game'].map(game_id_map)

#Converting Over======#

#delete row which has purchase value (because this is useless)
data = data[data['behavior name (purchase/play)'] != 'purchase']

# select only useful columns
selected_columns = ['User ID', 'Game ID', 'Hours']
filtered_data = data[selected_columns]


#//=========//
from sklearn.preprocessing import MinMaxScaler

def normalize(x):
    scaler = MinMaxScaler(feature_range=(0, 10))
    x = x.values.reshape(-1, 1)  #reshaping the dataframe for MinMaxScaler
    return scaler.fit_transform(x).flatten()

filtered_data['Rate'] = normalize(filtered_data['Hours'])
#//=========//

# Preprocessing over==========================================================#

from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np

# 데이터를 Surprise의 Dataset 형식으로 로드
reader = Reader(rating_scale=(0, 10))
ib_data = Dataset.load_from_df(filtered_data, reader)

# 아이템 기반 협업 필터링 모델 생성 (KNN 기반)
sim_options = {
  'name': 'cosine', # 아이템 간의 코사인 유사성을 사용
  'user_based': False # 아이템 기반 협업 필터링 사용
}
model = KNNBasic(sim_options=sim_options)

# 데이터를 학습용과 테스트용으로 분할
trainset, testset = train_test_split(ib_data, test_size=0.2)
model.fit(trainset)

# 예측 및 평가
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# print RMSE
print(f"RMSE: {rmse}")

# predict rating for a specific user and game
target_user_id = 151603712
target_game_id = 25
predicted_rating = model.predict(target_user_id, target_game_id)

# print predicted rating
print(predicted_rating)

# find game name corresponding to Game ID 25
game_id_to_find = 25
game_name = [game for game, idx in game_id_map.items() if idx == game_id_to_find]

# print corresponding game name if found
if game_name:
    print(f"The game corresponding to Game ID {game_id_to_find} is: {game_name[0]}")
else:
    print(f"No game found for Game ID {game_id_to_find}")
    