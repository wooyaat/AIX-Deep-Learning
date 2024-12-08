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
    print(f"Loaded user data: {len(data)} rows")  # 디버깅 출력
    return pd.DataFrame(data)

# 음식 데이터 로드
def load_food_data(filepath):
    """음식 데이터 로드"""
    food_data = pd.read_excel(filepath)
    food_data.fillna(0, inplace=True)  # 결측값 처리
    if 'id' not in food_data.columns:
        food_data['id'] = food_data.index  # 고유 ID 추가
    print(f"Loaded food data: {len(food_data)} rows")  # 디버깅 출력
    return food_data

# 데이터 증강 함수
def hierarchical_knn_augment(user_data, food_data, k=5, target_size=1000):
    """KNN을 활용한 계층적 데이터 증강"""
    combined_data = user_data.copy()

    num_missing = target_size - len(combined_data)
    if num_missing <= 0:
        return combined_data

    # KNN 학습용 데이터 준비
    food_data["Representative_Group"] = food_data["Representative Food Code"].astype(int)
    food_data["Subcategory_Group"] = food_data["Food Subcategory Code"].astype(int)
    feature_columns = ["Representative_Group", "Subcategory_Group", "calories", "protein", "fat", "carbs"]
    feature_array = food_data[feature_columns].to_numpy()

    # KNN 모델 학습
    knn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_model.fit(feature_array)

    new_data = []
    indices = np.random.choice(food_data.index, num_missing, replace=True)

    for idx in indices:
        query_point = feature_array[idx].reshape(1, -1)
        _, neighbors = knn_model.kneighbors(query_point)
        neighbor_indices = neighbors[0]

        # 이웃 데이터 혼합
        neighbor_data = food_data.iloc[neighbor_indices]
        numeric_data = neighbor_data.select_dtypes(include=[np.number])  # 숫자형 데이터만 선택
        mixed_data = numeric_data.mean(axis=0)  # 숫자형 데이터만 평균 계산

        # 노이즈 추가
        mixed_data["calories"] += np.random.normal(0, 10)
        mixed_data["protein"] += np.random.normal(0, 2)
        mixed_data["fat"] += np.random.normal(0, 1)
        mixed_data["carbs"] += np.random.normal(0, 3)

        # 추가적인 데이터 열 (id, liked 등) 설정
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
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        return self.models[arm].predict(X_scaled)

# JSON 데이터를 기반으로 테스트 데이터 생성
def generate_test_data_from_json(json_file):
    """
    user_data.json 파일에서 모든 사용자 데이터를 테스트 데이터로 반환
    """
    with open(json_file, "r") as f:
        user_data = json.load(f)

    test_data = []
    for user_id, user_info in user_data.items():
        recommendations = user_info.get("recommendations", [])
        for meal in recommendations:
            def safe_float(value):
                return float(value) if value not in ["nan", None, "", "-"] else 0.0

            test_data.append({
                "user_id": user_id,
                "food_name": meal.get("name", "Unknown"),
                "features": [
                    safe_float(meal.get("calories")),
                    safe_float(meal.get("protein")),
                    safe_float(meal.get("fat")),
                    safe_float(meal.get("carbs"))
                ]
            })

    print(f"Generated test data: {len(test_data)} items")  # 디버깅 출력
    return test_data

# 예측 및 상위 N개 출력
def predict_top_n_with_names(mab, test_features, food_data, top_n=3):
    predictions = []

    for arm in mab.arms:
        X_scaled = mab.scaler.transform(test_features)  # 스케일링 적용
        prediction = mab.predict(arm, X_scaled)
        predictions.append((arm, prediction[0][0]))

    # 확률 기준 정렬
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # 상위 N개의 팔 출력
    for i, (arm, prob) in enumerate(predictions[:top_n]):
        arm_data = food_data[food_data["Representative Food Code"] == arm]
        food_name = arm_data["name"].iloc[0] if not arm_data.empty else "Unknown"
        features = arm_data[["calories", "protein", "fat", "carbs"]].mean().values
        print(f"{i + 1}. 음식 이름: {food_name}, 특성: {features}, 좋아할 확률: {prob * 100:.2f}%")

# 실행
if __name__ == "__main__":
    # 사용자 및 음식 데이터 로드
    user_data = prepare_training_data(USER_DATA_FILE)
    food_data = load_food_data(FOOD_DB_FILE)

    # 데이터 증강
    augmented_data = hierarchical_knn_augment(user_data, food_data, target_size=1000)
    print(f"Augmented data size: {len(augmented_data)} rows")

    # 팔 정의
    arms = food_data["Representative Food Code"].unique()
    print(f"Defined arms: {len(arms)} unique codes")

    # MAB 모델 초기화
    input_shape = (4,)  # 칼로리, 단백질, 지방, 탄수화물
    mab = MultiArmedBandit(food_data, arms, input_shape)

    # 학습 루프
    for step in range(10):  # 학습 루프
        arm = mab.choose_arm(epsilon=0.1)  # 탐험/활용 결정
        arm_data = augmented_data[augmented_data["Representative Food Code"] == arm]
        if len(arm_data) == 0:
            print(f"No data for arm {arm}, skipping.")
            continue

        X = arm_data[["calories", "protein", "fat", "carbs"]].values
        y = arm_data["liked"].values

        # 학습
        mab.train_arm(arm, X, y)

        # 예측 결과를 기반으로 보상 설정
        predictions = mab.predict(arm, X)
        reward = int(np.round(np.mean(predictions)))  # 예측 평균을 반올림하여 보상 생성
        mab.update(arm, reward)

        # 디버깅 정보 출력
        loss = np.mean((predictions - y.reshape(-1, 1)) ** 2)
        print(f"Step {step + 1}: Trained on arm {arm}, Loss: {loss:.4f}, Reward: {reward}")


    # JSON 사용자 데이터를 기반으로 모든 테스트 데이터 처리
    test_data = generate_test_data_from_json(USER_DATA_FILE)
    print(f"Generated test data: {len(test_data)} items")

    for i, data in enumerate(test_data):
        user_id = data["user_id"]
        food_name = data["food_name"]
        test_features = np.array(data["features"]).reshape(1, -1)

        print(f"\n[{i + 1}] 사용자: {user_id}, 음식: {food_name}")
        predict_top_n_with_names(mab, test_features, food_data, top_n=3)
