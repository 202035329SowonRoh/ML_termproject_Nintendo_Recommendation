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


# print filtered_data
print(filtered_data)

#Preprocessing over==========================================================#

#Use Hours value as User's rate values.
#Because the higher values mean that user played more time on that game.
#So i decided to use Hours as the rate value.
#Scale Hours values into 0~10.0 value by using Min-Max scaling

from sklearn.preprocessing import MinMaxScaler

def min_max_scaling(x):
    scaler = MinMaxScaler(feature_range=(0, 1000)) #큰 의미가 없었는듯 scaling의 애초에 시간이 클수록 평가점수가 높은거니..
    x = x.values.reshape(-1, 1) #reshaping the dataframe for MinMaxScaler
    return scaler.fit_transform(x).flatten()

# Min max Scale the Hours value and put them into new Column 'Rate'
filtered_data['Rate'] = min_max_scaling(filtered_data['Hours'])

# select only useful columns
selected_columns2 = ['User ID', 'Game ID', 'Rate']
filtered_data = filtered_data[selected_columns2]

# drop rows where Rate is 0
filtered_data = filtered_data[filtered_data['Rate'] != 0]

# load data to Surprise's Dataset format
from surprise import Dataset, Reader
reader = Reader(rating_scale=(0, 10))
ib_data = Dataset.load_from_df(filtered_data, reader)

# item-based collaborative filtering model creation (KNN-based)
from surprise import KNNBasic
sim_options = {
    'name': 'cosine',  # use cosine similarity between items
    'user_based': False  # use item-based collaborative filtering
}
model = KNNBasic(sim_options=sim_options)

# use all data for training
trainset = ib_data.build_full_trainset()
model.fit(trainset)

# predict rating for a specific user and game
target_user_id = 151603712
target_game_id = 0
predicted_rating = model.predict(target_user_id, target_game_id)

# print predicted rating
print(predicted_rating)

# find game name corresponding to Game ID 25
game_id_to_find = 0
game_name = [game for game, idx in game_id_map.items() if idx == game_id_to_find]

# print corresponding game name if found
if game_name:
    print(f"The game corresponding to Game ID {game_id_to_find} is: {game_name[0]}")
else:
    print(f"No game found for Game ID {game_id_to_find}")




#Performance check RSME
#First, Test dataset is from the dataset
from surprise.model_selection import train_test_split
trainset2, testset = train_test_split(ib_data, test_size=0.2)

#uSE rsme TO calculate the performace
from surprise import accuracy

# RMSE 계산
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")