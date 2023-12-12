from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np



logging.basicConfig(level=logging.INFO)

app = FastAPI()

data_file_path = 'data.csv'

# 데이터 로드
df = pd.read_csv(data_file_path)

# 데이터 preprocessing
df = df.dropna(subset=['user_score']).reset_index(drop=True)
df = df.dropna(subset=['platform', 'esrb_rating', 'genres']).reset_index(drop=True)
# df = df.dropna(subset=['date'])

df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
df.dropna(subset=['year']).reset_index(drop=True)
df['year'] = df['year'].astype(int)


def content_based_filtering(df, user_profile):
    # 가중치 설정
    title_weight = 0.5
    genres_weight = 0.3
    esrb_rating_weight = 0.2
    
    # 'title'과 'genres'를 각각 TF-IDF 벡터로 변환
    title_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    title_tfidf_matrix = title_tfidf_vectorizer.fit_transform(df['title'])
    
    genres_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    genres_tfidf_matrix = genres_tfidf_vectorizer.fit_transform(df['genres'])
    
    
    #esrb 값을 문자열로 형 변환
    df['esrb_rating'] = df['esrb_rating'].astype(str)
    
    esrb_rating_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    esrb_rating_tfidf_matrix = esrb_rating_tfidf_vectorizer.fit_transform(df['esrb_rating'])
    
    # 각 속성에 대한 코사인 유사도 계산
    title_cosine_sim = cosine_similarity(title_tfidf_matrix, title_tfidf_matrix)
    genres_cosine_sim = cosine_similarity(genres_tfidf_matrix, genres_tfidf_matrix)
    esrb_rating_cosine_sim = cosine_similarity(esrb_rating_tfidf_matrix, esrb_rating_tfidf_matrix)
    
    # 사용자 프로파일과 가장 유사한 아이템
    item_index = df[df['title'] == user_profile['title']].index[0]
    
    # 각 속성별로 가중치를 곱한 유사도를 계산
    weighted_similarity = (title_cosine_sim[item_index] * title_weight +
                           genres_cosine_sim[item_index] * genres_weight +
                           esrb_rating_cosine_sim[item_index] * esrb_rating_weight)
    
    # 유사도 점수에 따라 정렬
    recommendations = []
    for idx, score in enumerate(weighted_similarity):
        if idx != item_index:  # 자기 자신은 제외
            recommendations.append((df.iloc[idx]['title'], df.iloc[idx]['user_score'], df.iloc[idx]['esrb_rating'], df.iloc[idx]['genres'], score))
    
    recommendations = sorted(recommendations, key=lambda x: x[4], reverse=True)
    
    # 상위 10개 추천만 반환
    recommendations = recommendations[:10]
    
    print(len(df))
    
    return recommendations

@app.get("/")
async def hello():
    logging.info("Hello, World!")
    return {"hello": "world!"}


# 컨텐츠 기반 필터링
@app.post("/recommend_games")
async def recommend_games(request: Request):
    try:
        # POST 메시지에서 JSON 데이터 추출
        request_data = await request.json()
        title = request_data.get("title", "")
        
        game_info = df[df['title'] == title].iloc[0]
        
        genres = game_info['genres']
        esrb = game_info['esrb_rating']
        
        user_profile = {
            'title' : title,
            'genres' : genres,
            'esrb_rating' : esrb
        }
        
        
        # 사용자 프로필을 기반으로 게임 추천 실행
        recommendations = content_based_filtering(df, user_profile)
        

        # 추천 결과를 JSON-serializable 형식으로 반환
        return JSONResponse(content=recommendations)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# 장르 json으로 입력하면 해당 장르에서 user score 높은 게임 5개 리턴
@app.post("/recommend_games_by_genre")
async def recommend_games_by_genre(request: Request):
    try:
        # POST 메시지에서 JSON 데이터 추출
        request_data = await request.json()
        genre = request_data.get("genre", "")

        # 입력된 장르에 해당하는 게임 필터링
        genre_filtered_df = df[df['genres'].str.lower().str.contains(genre.lower())]

        # 'user_score'로 정렬하여 상위 5개 게임 추출
        top_games_by_genre = genre_filtered_df.sort_values(by='user_score', ascending=False).head(5)

        # 추천 결과를 JSON-serializable 형식으로 반환
        recommendations = top_games_by_genre[['title', 'user_score', 'esrb_rating', 'genres']].to_dict(orient='records')

        return JSONResponse(content=recommendations)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# 입력한 ESRB와 같은 연령대 user score 높은 순으로 5개 리턴
@app.post("/recommend_games_by_esrb")
async def recommend_games_by_esrb_rating(request: Request):
    try:
        # POST 메시지에서 JSON 데이터 추출
        request_data = await request.json()
        esrb_rating = request_data.get("esrb_rating", "")

        # 입력된 ESRB 등급과 일치하는 게임 필터링
        esrb_filtered_df = df[df['esrb_rating'].str.lower() == esrb_rating.lower()]

        # 'user_score'로 정렬하여 상위 5개 게임 추출
        top_games_by_esrb_rating = esrb_filtered_df.sort_values(by='user_score', ascending=False).head(5)

        # 추천 결과를 JSON-serializable 형식으로 반환
        recommendations = top_games_by_esrb_rating[['title', 'user_score', 'esrb_rating', 'genres']].to_dict(orient='records')

        return JSONResponse(content=recommendations)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend_games_by_year")
async def recommend_games_by_year(request: Request):
    try:
        # POST 메시지에서 JSON 데이터 추출
        request_data = await request.json()
        request_year = request_data.get("date", "")

        # 입력된 연도와 일치하는 게임 필터링
        year_filtered_df = df[df['date'].str.contains(request_year)]

        # 'user_score'로 정렬하여 반환
        top_games_by_year = year_filtered_df.sort_values(by='user_score', ascending=False).head(5)

        # 추천 결과를 JSON-serializable 형식으로 반환
        recommendations = top_games_by_year[['title', 'user_score', 'esrb_rating', 'genres', 'date']].to_dict(orient='records')

        return JSONResponse(content=recommendations)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host='0.0.0.0', port=8000, workers=1)


#메모.. 
#dataset에 가중치를 주는것과 또 다른 방법으로 normalize하면 어땠을까요?
#null value값을 가진 data를 drop하지말고 평균값을 대입하는 방법처럼 다른 방법이 없을까요..
