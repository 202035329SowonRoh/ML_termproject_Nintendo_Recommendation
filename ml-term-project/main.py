import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

# # Streamlit에서 FastAPI 엔드포인트 호출하는 함수
# def get_recommendation(user_profile):
#     endpoint = f"{fastapi_url}/recommend_games"
#     response = requests.post(endpoint, json=user_profile)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error(f"Error: {response.text}")


# 데이터 로드
data = pd.read_csv('data.csv')

# 데이터 preprocessing
data = data.dropna(subset=['user_score']).reset_index(drop=True)
data = data.dropna(subset=['platform', 'esrb_rating', 'genres']).reset_index(drop=True)

# 'date' 열에서 년도 추출
# print(data['date'].head)

data['year'] = pd.to_datetime(data['date'], errors='coerce').dt.year
data.dropna(subset=['year']).reset_index(drop=True)
data['year'] = data['year'].astype(int)

# 고유한 년도 목록 생성
unique_years = sorted(data['year'].unique().tolist())

# 장르 분리, 목록 생성
unique_genres = set()
for genre_list in data['genres']:
    if isinstance(genre_list, str):
        # 대괄호와 따옴표 제거
        genre_list = genre_list.replace('[', '').replace(']', '').replace("'", "")
        genres = genre_list.split(', ')
        unique_genres.update(genres)

unique_genres = sorted(list(unique_genres))

# genres = data['genres'].unique().tolist()
esrb_ratings = data['esrb_rating'].unique().tolist() 


# Streamlit UI 생성
st.header(":red[Nintendo Switch Game] Recommendation", divider='rainbow')

titles = sorted(list(data['title']))

# 사이드바
st.sidebar.title("Filtering Options")

st.sidebar.subheader("Recommend games similar to the one you choose.")
st.sidebar.write("You can type and search for the game you want to find.")
selected_game = st.sidebar.selectbox("Select a Game", titles, index=None,
                                        placeholder="Select a Game")
                                        
st.sidebar.write("10 games are recommended based on your selection.")
st.sidebar.divider()

st.sidebar.subheader("Select recommendation option...")

# Disable the selectbox based on the value of selected_game
option = st.sidebar.selectbox("",['Genre','Year','ESRB'],index=None,
                                placeholder="Select a Option")
st.sidebar.write("5 games are recommended based on your selection.")


if option == 'Genre':
    selected_genre = st.sidebar.selectbox("Select Genre", unique_genres,index=None, placeholder="Select")
elif option == 'Year':
    selected_year = st.sidebar.selectbox("Recommend based on Year of Release", sorted(data['year'].unique().tolist()),
                                            index=None, placeholder="Select")
elif option == 'ESRB':
    text = "Recommend based on ESRB rating  \nESRB - A rating system used to determine the appropriate age group for game content."
    selected_esrb = st.sidebar.selectbox(text, esrb_ratings, index=None, placeholder="Select")

st.sidebar.divider()

# FastAPI 서버 엔드포인트 URL
fastapi_url = "http://3.38.143.123:8000"

# FastAPI 엔드포인트 호출, content-based 필터링
def get_recommendation(selected_game):
    endpoint = f"{fastapi_url}/recommend_games"
    # 선택하지 않은 경우 빈 문자열
    payload = {
        "title": selected_game
    }
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

# 장르 추천 astAPI 엔드포인트 호출
def get_recommendation_by_genre(selected_genre):
    endpoint = f"{fastapi_url}/recommend_games_by_genre"
    
    payload = {
        "genre": selected_genre
    }
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        print(response)
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

# esrb 추천 astAPI 엔드포인트 호출
def get_recommendation_by_esrb(selected_esrb):
    endpoint = f"{fastapi_url}/recommend_games_by_esrb"
    
    payload = {
        "esrb_rating": selected_esrb
    }
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        print(response)
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

# 년도별 추천 astAPI 엔드포인트 호출
def get_recommendation_by_year(selected_year):
    endpoint = f"{fastapi_url}/recommend_games_by_year"
    
    payload = {
        "date": str(selected_year)
    }
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        print(response)
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None

opcode = 'none'

# 추천 게임 표시
if st.sidebar.button("Give Me Recommendation!"):
    
    #
    if option == 'Genre':
        opcode = 'genre'
        print(opcode)
        recommendations = get_recommendation_by_genre(selected_genre)
    elif option == 'Year':
        opcode = 'year'
        print(opcode)
        recommendations = get_recommendation_by_year(selected_year)
    elif option == 'ESRB':
        opcode = 'esrb'
        print(opcode)
        recommendations = get_recommendation_by_esrb(selected_esrb)
    else:
        opcode = 'content'
        print(opcode)
        print(len(data))
        recommendations = get_recommendation(selected_game)
        
    with st.expander("Show the Result Graph"):
        result_graph = st.container()
    
    recommend_titles = []
    recommend_scores = []
    
    if recommendations:
        st.write("Recommended Games:")
        for game in recommendations:
            if (opcode == 'content'):
                title = game[0]  # 게임 타이틀은 첫 번째 요소
                score = game[-1]  # score는 마지막 요소
                #st.text(f"Title: {title}\nScore: {score}")
                
                # score 소수점 2자리 까지
                formatted_score = "{:.2f}".format(float(score))
                
                recommend_titles.append(title)
                recommend_scores.append(formatted_score)
                
                recommend_result = st.container(border=True)
                col1, col2 = recommend_result.columns([2,1])
                with col1:
                    st.header(title)
                    st.link_button("Google this game","https://www.google.com/search?q="+title)
                with col2:
                    st.text("Recommendation Score")
                    st.subheader(formatted_score)
            else :
                #st.text(f"Title: {game['title']} \nUser Score: {game['user_score']}")
                
                title = game['title']
                score = game['user_score']
                
                recommend_titles.append(title)
                recommend_scores.append(score)
                
                recommend_result = st.container(border=True)
                with recommend_result:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.header(title)
                        st.link_button("Google this game","https://www.google.com/search?q="+title)
                    with col2:
                        st.text("Average User Score")
                        st.subheader(score)
                
    else:
        st.warning("No recommendations found.")

    with result_graph:
        
        recommend_titles.reverse()
        recommend_scores.reverse()
        
        # Create a bar plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        bar_colors = ['#b1cefa'] * (len(recommend_titles) - 1) + ['#f55142']
        ax.bar(recommend_titles, recommend_scores, color=bar_colors, width=0.5)
        
        # Set x-axis ticks and rotate labels
        ax.set_xticks(recommend_titles)
        ax.set_xticklabels(recommend_titles, rotation=30, ha="right", rotation_mode="anchor")
        
        # Customize the plot
        ax.set_xlabel("Game Title")
        ax.set_ylabel("Recommendation Score")
        ax.set_title("Recommended Games")
        
        # Show the plot in Streamlit app
        st.pyplot(fig)
    