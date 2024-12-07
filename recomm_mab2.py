import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# TensorFlow 즉시 실행 활성화 및 디버그 모드 설정
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# 파일 경로
FOOD_DB_FILE = "food_db2.xlsx"
USER_DATA_FILE = "user_data.json"

# 사용자 데이터 준비
def prepare_training_data(filepath):
    """user_data.json 파일에서 데이터를 로드하고 학습 데이터를 생성"""
    with open(filepath, "r") as f:
        user_data = json.load(f)

    data = []
    for user_id, user_info in user_data.items():
        for meal in user_info.get("recommendations", []):
            data.append({
                "user_id": user_id,
                "calories": float(meal.get("calories", 0)),
                "protein": float(meal.get("protein", 0)),
                "fat": float(meal.get("fat", 0)),
                "carbs": float(meal.get("carbs", 0)),
                "liked": 1  # 추천 데이터는 기본적으로 좋아한다고 가정
            })
    return pd.DataFrame(data)

# 음식 데이터 로드
def load_food_data(filepath):
    """음식 데이터 로드"""
    food_data = pd.read_excel(filepath)
    if 'id' not in food_data.columns:
        food_data['id'] = food_data.index  # 고유 ID 추가
    return food_data[["id", "Representative Food Code", "Food Subcategory Code", "calories", "protein", "fat", "carbs"]]

# 데이터 증강 함수
def hierarchical_knn_augment(user_data, food_data, k=5, target_size=1000):
    """KNN을 활용한 계층적 데이터 증강"""
    combined_data = user_data.copy()

    # 결측값 처리: 0으로 대체
    food_data.fillna(0, inplace=True)

    num_missing = target_size - len(combined_data)
    if num_missing <= 0:
        return combined_data

    food_data["Representative_Group"] = food_data["Representative Food Code"].astype(int)
    food_data["Subcategory_Group"] = food_data["Food Subcategory Code"].astype(int)
    feature_columns = ["Representative_Group", "Subcategory_Group", "calories", "protein", "fat", "carbs"]

    # `numpy` 형식으로 변환
    feature_array = food_data[feature_columns].to_numpy()

    # KNN 모델 학습
    knn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_model.fit(feature_array)

    new_data = []
    indices = np.random.choice(food_data.index, num_missing, replace=True)

    for idx in indices:
        # 입력 데이터도 `numpy` 배열로 변환
        query_point = feature_array[idx].reshape(1, -1)
        _, neighbors = knn_model.kneighbors(query_point)
        neighbor_indices = neighbors[0]

        # 이웃 데이터 혼합
        mixed_data = food_data.iloc[neighbor_indices].mean(axis=0)
        mixed_data["calories"] += np.random.normal(0, 10)
        mixed_data["protein"] += np.random.normal(0, 2)
        mixed_data["fat"] += np.random.normal(0, 1)
        mixed_data["carbs"] += np.random.normal(0, 3)
        mixed_data["liked"] = 1  # 증강된 데이터는 선호 데이터로 가정
        new_data.append(mixed_data)

    new_data_df = pd.DataFrame(new_data)
    combined_data = pd.concat([combined_data, new_data_df], ignore_index=True)
    return combined_data

# MAB 클래스 정의
class MultiArmedBandit:
    def __init__(self, food_data, arms, input_shape):
        self.arms = arms
        self.models = {arm: self.build_model(input_shape) for arm in arms}
        self.rewards = {arm: 0 for arm in arms}
        self.counts = {arm: 0 for arm in arms}
        self.scaler = StandardScaler()
        self.food_data = food_data

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = Dense(256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)

        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)

        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def choose_arm(self, epsilon=0.1):
        if np.random.rand() < epsilon:  # 탐험
            return np.random.choice(self.arms)
        else:  # 활용
            avg_rewards = {arm: self.rewards[arm] / (self.counts[arm] + 1e-6) for arm in self.arms}
            return max(avg_rewards, key=avg_rewards.get)

    def update(self, arm, reward):
        self.rewards[arm] += reward
        self.counts[arm] += 1

    def train_arm(self, arm, X, y, epochs=10, batch_size=32):
        X_scaled = self.scaler.fit_transform(X)
        self.models[arm].fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, arm, X):
        """명시적으로 numpy 입력 처리"""
        X_scaled = self.scaler.transform(X)
        return self.models[arm].predict(X_scaled)

# 예측 및 상위 N개 출력
def predict_top_n(mab, test_features, top_n=3):
    """MAB 모델에서 테스트 데이터를 기반으로 상위 N개의 팔과 확률 출력"""
    predictions = []

    for arm in mab.arms:
        prediction = mab.predict(arm, test_features)
        predictions.append((arm, prediction[0][0]))  # 팔 번호와 확률 저장

    # 확률 기준으로 정렬
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # 상위 N개의 팔과 확률 출력
    print(f"상위 {top_n}개의 팔과 예측 확률:")
    for i, (arm, prob) in enumerate(predictions[:top_n]):
        print(f"{i + 1}. 팔: {arm}, 좋아할 확률: {prob * 100:.2f}%")

# 실행
if __name__ == "__main__":
    # 사용자 및 음식 데이터 로드
    user_data = prepare_training_data(USER_DATA_FILE)
    food_data = load_food_data(FOOD_DB_FILE)

    # 데이터 증강
    augmented_data = hierarchical_knn_augment(user_data, food_data, target_size=1000)

    # 팔 정의
    arms = food_data["Representative Food Code"].unique()

    # MAB 모델 초기화
    input_shape = (4,)  # 칼로리, 단백질, 지방, 탄수화물
    mab = MultiArmedBandit(food_data, arms, input_shape)

    # 학습 루프
    for step in range(100):
        arm = mab.choose_arm(epsilon=0.1)
        arm_data = augmented_data[augmented_data["Representative Food Code"] == arm]
        if len(arm_data) == 0:
            continue  # 해당 팔에 데이터가 없으면 스킵
        X = arm_data[["calories", "protein", "fat", "carbs"]].values
        y = arm_data["liked"].values
        reward = np.random.choice([0, 1], p=[1 - y.mean(), y.mean()])  # 보상
        mab.train_arm(arm, X, y)
        mab.update(arm, reward)

    # 예측 테스트
    test_features = np.array([[500, 25, 10, 60]])  # 예시 데이터
    test_features = test_features.reshape(-1, 4)   # 항상 일정한 크기로 변환
    predict_top_n(mab, test_features, top_n=3)  # 상위 3개 출력
