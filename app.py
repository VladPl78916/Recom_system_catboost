import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger

app = FastAPI()

def batch_load_sql(query: str):
    engine = create_engine(
        "postgresql://your_database"
        "postgres.lab.your_database"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_features():
    # Уникальные записи post_id, user_id, где был совершён лайк
    logger.info("loading liked posts")

    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    
    liked_posts = batch_load_sql(liked_posts_query)

    # Фичи по постам на основе tf_idf
    logger.info("loading posts features")
    posts_features = pd.read_sql(
        """SELECT * FROM public.vpd_658_posts_info_features""",

        con ="postgresql://your_database"
             "postgres.lab.your_database"
    )

    # Фичи по юзерам
    logger.info("loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",

        con ="postgresql:your_database"
             "postgres.lab.your_database"
    )

    return [liked_posts, posts_features, user_features]

def load_models():
    # Используем get_model_path для получения пути к модели
    model_path = get_model_path("catboost_model")
    
    # Создаем экземпляр CatBoostClassifier и загружаем модель
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    
    # Возвращаем загруженную модель
    return loaded_model

# Положим при поднятии сервиса модель и фичи в переменные model, features

logger.info("loading model")
model = load_models()
logger.info("loading features")
features = load_features()
logger.info("service is up and running")

def get_recommended_feed(id: int, time: datetime, limit:int):
    # загрузим фичи по пользователям
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop('user_id', axis=1)

    # Загрузим фичи по постам
    logger.info("dropping columns")
    posts_features = features[1].drop(['index', 'text'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    # Объединим эти фичи
    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Добавим информацию о дате рекомендаций
    logger.info("add time info")
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    # Сформируем предсказания вероятности дайкнуть пост для всех постов
    logger.info("predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Уберём записи, где пользователь ранее уже ставил лайк
    logger.info("deleting liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-5 по вероятности постов
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int, 
		time: datetime, 
		limit: int = 10) -> List[PostGet]:
    return get_recommended_feed(id, time, limit)