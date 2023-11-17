from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import mysql.connector
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from collections import defaultdict
from flask_caching import Cache
import threading
from concurrent.futures import ThreadPoolExecutor
from mysql.connector.pooling import MySQLConnectionPool
from config import config

app = Flask(__name__) # Flask 애플리케이션 객체 생성
CORS(app)  # 크로스 도메인 요청 허용을 위한 CORS 설정

# 캐시 설정을 SimpleCache로 초기화합니다.
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# 로그 설정을 INFO 레벨로 초기화하고 포맷을 설정합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

# 데이터베이스 연결 풀을 생성합니다. 최대 10개의 연결을 허용합니다.
cnxpool = MySQLConnectionPool(pool_name="mypool", pool_size=10, **config)


# ---------------------------- DB에 관련된 함수 ----------------------------

# 연결 풀에서 커넥션 및 커서를 가져오는 함수
def get_cursor_and_connection_from_pool():
    connection = cnxpool.get_connection()
    return connection.cursor(), connection

# 데이터베이스에서 영화 정보를 가져오는 함수.
@cache.memoize(timeout=3600)  # 1시간 동안 캐시 유지
def fetch_movie_data():
    cursor, connection = get_cursor_and_connection_from_pool()
    cursor.execute("SELECT movie_id, us_title, kr_title, overview, genres, directors, actors FROM movie_info")
    movie_data = cursor.fetchall() #쿼리 결과를 모두 가져옴
    connection.close()
    return movie_data

# 초기 영화 데이터 로딩 후, DataFrame 형태로 반환하는 함수
def initialize_movie_data():
    movie_data = fetch_movie_data()
    columns = ["movie_id", "us_title", "kr_title", "overview", "genres", "directors", "actors"]

    # DateFrame 생성
    df = pd.DataFrame(movie_data, columns=columns)
    df['actors'] = df['actors'].apply(lambda x: ",".join(x.split(",")[:3])) # 배우 이름을 ','로 구분하고, 처음 3명의 배우 불러옴
    return df

# 사용자가 본 영화의 ID를 반환하는 함수
def fetch_rated_movies(user_email):
    cursor, connection = get_cursor_and_connection_from_pool()
    cursor.execute("SELECT movie_id FROM score WHERE user_email = %s", (user_email,))
    movies = cursor.fetchall()
    connection.close()
    return [movie[0] for movie in movies] # 사용자가 본 영화의 ID를 리스트 형태로 반환

# 데이터베이스에서 평점 데이터를 로드하는 함수
def load_data_from_db():
    cursor, connection = get_cursor_and_connection_from_pool()
    if not cursor or not connection:
        return pd.DataFrame()
    try:
        cursor.execute("SELECT user_email, movie_id, score FROM score")
        ratings = cursor.fetchall()
        return pd.DataFrame(ratings, columns=['user_email', 'movie_id', 'score'])
    except Exception as e:
        logging.error(f"데이터 로드 중 오류: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

# 이전에 저장된 추천 결과를 삭제하는 함수
def delete_previous_suggestions(user_email):
    cursor, connection = get_cursor_and_connection_from_pool()
    if not cursor or not connection:
        return
    try:
        delete_query = "DELETE FROM suggestion WHERE user_email = %s"
        cursor.execute(delete_query, (user_email,))
        connection.commit()
    except Exception as e:
        logging.error(f"영화 추천 정보 삭제 중 오류: {e}")
    finally:
        connection.close()

# 추천 결과를 데이터베이스에 저장하는 함수
def bulk_insert_into_suggestions(movies, user_email):
    connection = None
    try:
        # 데이터베이스 연결
        connection = cnxpool.get_connection()
        cursor = connection.cursor()

        # INSERT 쿼리를 정의
        insert_query = "INSERT INTO suggestion (movie_id, user_email) VALUES (%s, %s)"

        # 영화 ID와 사용자 이메일로 구성된 값들의 리스트를 생성
        values = [(int(movie_id), user_email) for movie_id in movies]

        # executemany를 사용하여 여러 개의 데이터를 한 번에 데이터베이스에 삽입
        cursor.executemany(insert_query, values)
        connection.commit()

    except mysql.connector.Error as e:
        logging.error(f"추천 영화 저장 중 오류: {e}")
        if connection:
            connection.rollback()  # 오류 발생 시 롤백 처리

    finally:
        if connection:
            connection.close()

# 사용자가 가장 최근에 3점 이상을 준 영화를 찾는 함수
def get_recent_high_rated_movie(user_email):
    cursor, connection = get_cursor_and_connection_from_pool()
    select_query = """
    SELECT movie_id 
    FROM score 
    WHERE user_email = %s AND score >= 3 
    ORDER BY times DESC
    LIMIT 1
    """
    cursor.execute(select_query, (user_email,))
    movie_data = cursor.fetchone()
    connection.close()

    if movie_data:
        return movie_data[0]
    return None





# ---------------------------- 컨텐츠 기반 추천 관련 함수 ----------------------

# 텍스트를 토큰화하는 함수(감독, 배우에서 사용)
def tokenize(text):
    return [word.strip() for word in text.split(',')]

# 사용자의 선호도로부터 가중치를 적용하는 함수
def get_weights_from_preferences(preference_1, preference_2, preference_3):
    # 초기 가중치
    weights = {
        'overview': 0,
        'actors': 0,
        'genres': 0,
        'directors': 0
    }

    preferences = [preference_1, preference_2, preference_3]
    preference_weights = [30, 20, 10]
    for i, pref_type in enumerate(preferences):
        if pref_type == 1:
            weights['directors'] += preference_weights[i]
        elif pref_type == 2:
            weights['genres'] += preference_weights[i]
        elif pref_type == 3:
            weights['actors'] += preference_weights[i]

    return weights

# 사용자 선호도에 대한 가중치를 반환하는 함수
def get_user_preference_weights(user_email):
    cursor, connection = get_cursor_and_connection_from_pool()
    select_query = "SELECT preference_1, preference_2, preference_3 FROM users WHERE email = %s"
    cursor.execute(select_query, (user_email,))
    user_data = cursor.fetchone()
    connection.close()

    if not user_data:
        raise ValueError('No user found with provided email')

    preference_1, preference_2, preference_3 = user_data
    return get_weights_from_preferences(preference_1, preference_2, preference_3)

# 주어진 컬럼과 가중치로 TF-IDF(Term Frequency-Inverse Document Frequency)벡터화를 수행하는 함수
def weighted_vectorize(column_name, weight):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_name].fillna(''))
    return tfidf_matrix * weight

# 가중치를 바탕으로 코사인 유사도 행렬을 계산하는 함수
@cache.memoize(timeout=3600)  # 1시간 동안 캐시 유지
def compute_cosine_sim_from_weights(weights):
    overview_matrix = weighted_vectorize("overview", weights['overview'])
    actors_matrix = weighted_vectorize("actors", weights['actors'])
    genres_matrix = weighted_vectorize("genres", weights['genres'])
    directors_matrix = weighted_vectorize("directors", weights['directors'])

    # 행렬 결합
    combined_matrix = hstack([overview_matrix, actors_matrix, genres_matrix, directors_matrix])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(combined_matrix)

    return combined_matrix, cosine_sim

# 주어진 영화와 유사한 영화를 추천
def recommend_similar_movies(movie_id, data_frame, cosine_sim_matrix, user_email, top_n=10):
    def get_movie_details_from_index(index):
        return (
            data_frame.loc[index, "movie_id"],
            data_frame.loc[index, "us_title"],
            data_frame.loc[index, "kr_title"],
            data_frame.loc[index, "overview"],
            data_frame.loc[index, "genres"],
            data_frame.loc[index, "directors"],
            data_frame.loc[index, "actors"]
        )

    def get_index_from_movie_id(movie_id):
        matching_rows = data_frame[data_frame.movie_id == movie_id]
        if matching_rows.shape[0] > 0:
            return matching_rows.index[0]
        else:
            return None

    # 사용자가 평가한 영화의 ID를 가져옵니다.
    rated_movies = fetch_rated_movies(user_email)

    # 평점이 주어진 영화를 전체 데이터프레임에서 제외합니다.
    remaining_movies = data_frame[~data_frame.movie_id.isin(rated_movies)]
    new_indices = remaining_movies.index

    # 제외한 후의 새로운 인덱스를 얻습니다.
    movie_index = get_index_from_movie_id(movie_id)
    if movie_index is None:
        return None, []

    # 영화 간의 유사도 계산(주어진 movie_id의 영화와 다른 모든 영화간의 코사인 유사도를 계산하고 내림차순 정렬)
    similar_movies = list(enumerate(cosine_sim_matrix[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    # 유사한 영화 목록을 만듭니다.(평가되지 않은 영화 중에서 수행)
    recommended_movie_indices = [movie[0] for movie in sorted_similar_movies if movie[0] in new_indices][:top_n]

    # 추천된 영화 세부정보
    input_movie_details = get_movie_details_from_index(movie_index)
    recommended_movies = [get_movie_details_from_index(i) for i in recommended_movie_indices]

    return input_movie_details, recommended_movies



# ---------------------------- 사용자 기반 추천 관련 함수 -------------------------

# 주어진 사용자가 학습 데이터셋에 있는지 확인하는 함수
def user_in_trainset(user_email, trainset):
    try:
        _ = trainset.to_inner_uid(user_email)
        return True
    except ValueError:
        return False

# KNN 모델 학습 함수
def train_knn_model(ratings_df):

    # 평점 데이터 범위 지정(1~5점)
    reader = Reader(rating_scale=(1, 5))

    # 데이터프레임에서 surprise 라이브러리가 사용할 수 있는 데이터셋으로 변환
    data = Dataset.load_from_df(ratings_df[['user_email', 'movie_id', 'score']], reader)

    # 데이터를 훈련 세트와 테스트 세트로 분할 (80:20 비율)
    trainset, testset = train_test_split(data, test_size=0.2)

    # KNN 모델 정의 및 훈련
    algo = KNNWithMeans(k=10, sim_options={'name': 'cosine', 'user_based': True, 'min_support': 3})
    algo.fit(trainset)  # 훈련 세트로 알고리즘 학습

    return algo

# 주어진 사용자에 대한 영화 추천을 반환하는 함수
def get_movie_recommendations(user_email, n=10):

    # 사용자가 평점을 준 영화
    rated_movies = fetch_rated_movies(user_email)

    # 사용자의 내부 ID 가져오기
    inner_uid = algo.trainset.to_inner_uid(user_email)

    # 이웃 찾기
    neighbors = algo.get_neighbors(inner_uid, k=10)

    # 이웃의 ID 가져오기
    neighbor_uids = [algo.trainset.to_raw_uid(inner_id) for inner_id in neighbors]

    # 각 영화 ID에 대한 이웃 사용자들의 평점을 저장할 것
    scores_from_neighbors = defaultdict(list)
    for neighbor_uid in neighbor_uids:
        inner_id_neighbor = algo.trainset.to_inner_uid(neighbor_uid)
        inner_uid = algo.trainset.to_inner_uid(user_email)
        similarity_score = algo.sim[inner_uid][inner_id_neighbor]

        neighbor_ratings = ratings_df[ratings_df['user_email'] == neighbor_uid]
        for index, row in neighbor_ratings.iterrows():
            movie_id, score = row['movie_id'], row['score']
            if movie_id not in rated_movies:
                scores_from_neighbors[movie_id].append(score)

    # 수집한 평점을 기준으로 각 영화에 대한 평균 점수를 계산하고 내림차순 정렬
    avg_scores = {movie_id: sum(scores) / len(scores) for movie_id, scores in scores_from_neighbors.items()}
    sorted_recommendations = sorted(avg_scores, key=avg_scores.get, reverse=True)

    return sorted_recommendations[:n]



# ---------------------------- 알고리즘 수행 함수 --------------------------

# 컨텐츠 기반 알고리즘 실행 함수
def contents_algorithm_recommendation(user_email, movie_id, weights):
    # 사용자의 선호도 가져오기
    weights = get_user_preference_weights(user_email)

    # 코사인 유사도 행렬을 계산 결과 가져오기
    combined_matrix, cosine_sim = compute_cosine_sim_from_weights(weights)

    # 영화 추천 결과 가져오기
    input_movie_details, similar_movies = recommend_similar_movies(movie_id, df, cosine_sim, user_email, top_n=10)

    # 추천된 영화 ID 목록 구성
    recommended_movie_ids = [movie[0] for movie in similar_movies]

    # 이전 추천 삭제
    delete_previous_suggestions(user_email)

    # DB에 영화 추천 결과 넣기
    bulk_insert_into_suggestions(recommended_movie_ids, user_email)

    return [movie[0] for movie in similar_movies]

# 하이브리드 알고리즘 실행 함수
def hybrid_recommendation(user_email, movie_id):
    # 사용자의 평점 데이터 수를 얻습니다.
    user_ratings_count = len(ratings_df[ratings_df['user_email'] == user_email])

    # 해당 사용자와 유사도가 0.5 이상인 사용자 수를 확인합니다.
    inner_uid = algo.trainset.to_inner_uid(user_email)
    user_similarities = algo.sim[inner_uid]
    similar_users_count = sum(1 for sim_score in user_similarities if sim_score >= 0.5)

    # 사용자 평점 데이터 수나 유사한 사용자 수가 충분하지 않은 경우 컨텐츠 기반 추천으로 전환합니다.
    if user_ratings_count < 5 or similar_users_count < 5:
        contents_algorithm_recommendation(user_email, movie_id, get_user_preference_weights(user_email))
        return

    # 추천 결과를 저장할 딕셔너리 생성
    combined_recommendations = defaultdict(int)

    # 병렬 작업을 수행하기 위해 ThreadPoolExecutor 사용
    with ThreadPoolExecutor(max_workers=2) as executor:

        # 사용자 기반 추천 및 컨텐츠 기반 추천을 병렬로 실행
        user_based_future = executor.submit(get_movie_recommendations, user_email, 10)
        content_based_future = executor.submit(contents_algorithm_recommendation, user_email, movie_id, False)

        # 병렬 작업 결과
        user_based_recommendations = user_based_future.result()
        content_based_recommendations = content_based_future.result()

    # 추천 결과를 결합(컨텐츠 5개, 사용자 5개)
    for i, (user_movie, content_movie) in enumerate(zip(user_based_recommendations, content_based_recommendations)):
        score = 10 - i
        combined_recommendations[user_movie] += score
        combined_recommendations[content_movie] += score

    # 하이브리드 최종 추천 결과
    final_recommendations = sorted(combined_recommendations, key=combined_recommendations.get, reverse=True)[:10]

    # 이전 추천 삭제
    delete_previous_suggestions(user_email)

    # 영화 추가
    bulk_insert_into_suggestions(final_recommendations, user_email)




# ---------------------------- Flask 연동 -----------------------
# 추천시 영화 추천해주는 코드
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.json
    input_movie_id = int(data.get('movie_id'))
    user_email = data.get('user_email')
    rating = float(data.get('score', 0))

    # ratings_df 갱신
    global ratings_df
    ratings_df = load_data_from_db()

    # KNN 모델 재학습
    global algo
    algo = train_knn_model(ratings_df)

    # 유저가 trainset에 없다면 KNN 모델 재학습
    if not user_in_trainset(user_email, algo.trainset):
        algo = train_knn_model(ratings_df)

    # 평점 3점 이상주었다면 영화 추천 시작
    if rating >= 3:
        hybrid_recommendation(user_email, input_movie_id)
    else:
        # 사용자의 가장 최근에 준 3점 이상의 평점 영화 찾기
        recent_high_rated_movie_id = get_recent_high_rated_movie(user_email)
        if recent_high_rated_movie_id:
            hybrid_recommendation(user_email, recent_high_rated_movie_id)

    return jsonify({"message": "평점이 성공적으로 처리되었습니다."})

# 사용자가 선호도를 변경했을때 수행되는 함수
@app.route('/change', methods=['POST'])
def change_movies():
    data = request.json
    user_email = data.get('user_email')

    global ratings_df
    ratings_df = load_data_from_db()

    global algo
    algo = train_knn_model(ratings_df)

    if not user_in_trainset(user_email, algo.trainset):
        algo = train_knn_model(ratings_df)

    # 가장 최근에 평점 3점 이상 준거로 하이브리드 알고리즘 수행(선호도가 변경되었으니 변경되는점이 있을 것)
    recent_high_rated_movie_id = get_recent_high_rated_movie(user_email)
    hybrid_recommendation(user_email, recent_high_rated_movie_id)

    return jsonify({"message": "변경된 선호도로 영화 추천이 처리되었습니다."})

# 유저 회원가입시 유저 이메일을 받아 초기 알고리즘 작업 코드
@app.route('/userjoin', methods=['POST'])
def register_and_recommend():
    data = request.json
    user_email = data.get('user_email')

    print(user_email)

    # 사용자 이메일로 users 테이블에서 선호도 정보 가져오기
    cursor, connection = get_cursor_and_connection_from_pool()
    select_query = "SELECT movie_id, preference_1, preference_2, preference_3 FROM users WHERE email = %s"
    cursor.execute(select_query, (user_email,))
    user_data = cursor.fetchone()
    connection.close()

    if not user_data:
        return jsonify({'error': 'No user found with provided email'}), 400

    movie_id, preference_1, preference_2, preference_3 = user_data

    weights = get_weights_from_preferences(preference_1, preference_2, preference_3)
    combined_matrix, cosine_sim = compute_cosine_sim_from_weights(weights)

    # 입력받은 favorite_movie_id를 기반으로 추천 영화 목록 생성
    input_movie_details, similar_movies = recommend_similar_movies(movie_id, df, cosine_sim, user_email, top_n=10)

    # similar_movies에서 movie_id와 kr_title을 추출
    similar_movies_details = [(int(movie[0]), movie[2]) for movie in similar_movies]

    movie_ids_to_insert = [movie[0] for movie in similar_movies_details]
    bulk_insert_into_suggestions(movie_ids_to_insert, user_email)

    return jsonify({'status': 'success', 'recommended_movies': similar_movies_details})


# 메인 실행부
if __name__ == '__main__':
    # 영화 데이터 초기화
    df = initialize_movie_data()

    # ratings_df 로드
    ratings_df = load_data_from_db()

    app.run(debug=True, host='0.0.0.0', port=5100, threaded=True)