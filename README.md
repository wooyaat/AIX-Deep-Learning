# AIX-Deep-Learning
# Title : 1인 가구를 위한 사용자 맞춤형 식단 추천 알고리즘
# Members 
  이호원, 전기공학과, howonlee1217@gmail.com / 박도연, 수학과, nsa09041@naver.com
# Proposal(Option A)
 - Motivation :
 - Goal : 사용자 정보를 기반으로 하루 권장 칼로리를 계산 후 추천 식단을 제공
# Datasets
 - 식품의약품안전처 식품 영양성분 데이터베이스
 - https://various.foodsafetykorea.go.kr/nutrient/general/down/list.do
 - <img width="1469" alt="image" src="https://github.com/user-attachments/assets/d95f9afa-ea5b-4900-b091-1759f286c084">

# Methodology
recomm.py
1. **칼로리 계산**: 사용자의 기본 정보를 바탕으로 하루 권장 칼로리를 계산합니다.
2. **식단 추천**: 권장 칼로리 범위 내에서 적합한 음식을 추천합니다.
3. **데이터 저장**: 사용자 입력 및 추천 결과를 JSON 파일에 저장합니다.
---
## **코드 주요 부분**

### **0. 코드 실행에 필요한 라이브러리와 모듈 준비
```python
import pandas as pd
### -> food_db2.xlsx 파일을 처리하기 위해 사용합니다.

import json
### -> user_data.json 파일을 생성해 사용자 데이터를 구축하기 위해 사용합니다.

import os
### -> user_data.json 파일과 관련된 작업을 진행하기 위해 사용합니다.
### -> 구체적으로 파일의 존재 여부를 파악한 뒤 있다면 내용을 읽어오고 없다면 초기화된 데이터를 사용할 수 있도록 합니다.

### **1. 데이터 로드**
file_path = 'food_db2.xlsx'
food_data = pd.read_excel(file_path)

if 'id' not in food_data.columns:
    food_data['id'] = food_data.index

### -> food_db2.xlsx에서 음식 데이터를 불러오고, 고유 ID를 추가합니다.

### **2. 사용자 입력 받기**
def get_user_input():
    gender = input("성별을 입력하세요 (male/female): ").strip().lower()
    age = int(input("나이를 입력하세요: "))
    weight = float(input("몸무게를 입력하세요 (kg): "))
    height = float(input("키를 입력하세요 (cm): "))
    activity_level = input("활동 수준을 선택하세요 (sedentary/lightly_active/moderately_active/very_active/super_active): ").strip()
    meal_count = int(input("하루 두 끼(2) 또는 세 끼(3) 중 선택하세요: "))
    return gender, age, weight, height, activity_level, meal_count

### -> 사용자가 성별, 나이, 몸무게, 활동 수준 등을 입력하면 이 데이터를 반환합니다.

### **3. 하루 권장 칼로리 계산**
def calculate_daily_calories(gender, age, weight, height, activity_level):
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_multiplier = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "super_active": 1.9,
    }
    return bmr * activity_multiplier.get(activity_level, 1.2)

### -> 사용자의 성별, 나이, 몸무게, 키를 기준으로 **기초 대사량(BMR)**을 계산합니다.
### -> activity_level에 따라 활동 계수를 곱해 최종 권장 칼로리를 산출합니다.

### **4. 식단 추천**
def recommend_meals(food_data, meal_calories, exclude_ids, number=3):
    filtered_data = food_data[
        (food_data['calories'] <= meal_calories) &
        (~food_data['id'].isin(exclude_ids))
    ].sort_values(by='calories', ascending=False).head(number)
    return filtered_data

### -> 권장 칼로리를 초과하지 않는 음식 데이터를 필터링합니다.
### -> 칼로리가 높은 순으로 정렬하여 추천합니다.

### **5. 사용자 데이터 저장**
def save_user_data(user_id, input_data, recommendations):
    user_data_file = "user_data.json"
    if os.path.exists(user_data_file):
        with open(user_data_file, "r") as f:
            user_data = json.load(f)
    else:
        user_data = {}

    recommendations_dict = recommendations.astype(str).to_dict(orient="records")

    user_data[user_id] = {
        "input_data": input_data,
        "recommendations": recommendations_dict
    }

    with open(user_data_file, "w") as f:
        json.dump(user_data, f, indent=4)
    print("사용자 데이터가 저장되었습니다.")

### -> 추천 결과와 사용자 데이터를 JSON 파일로 저장하여 재사용 가능하도록 합니다.
```

recomm_ㅡmab2.py
1. **데이터 로드 및 전처리**: 음식 데이터와 사용자 데이터를 읽어오고 학습 가능한 형태로 준비합니다.
2. **데이터 증강**: 권장 칼로리 범위 내에서 적합한 음식을 추천합니다.
3. **Multi-Armed Bandit(MAB)**: 각 음식 그룹(팔)을 학습하고 보상 기반으로 최적의 음식을 선택합니다.
4. **추천 출력**: 사용자가 선호할 가능성이 높은 음식 상위 3개를 예측하여 제공합니다.
---
## **코드 주요 부분**

### **0. 코드 실행에 필요한 라이브러리와 모듈 준비
```python
import json
### -> user_data.json 파일을 통해 사용자 데이터를 저장하고 읽어오기 위해 사용합니다.

import numpy as np
### -> KNN 기반 데이터 증강에서 랜덤 샘플링과 수치 연산 수행을 위해 사용합니다.

import pandas as pd
### -> json을 통해 읽어온 사용자 데이터와 food_db2.xlsx 파일을 처리하기 위해 사용합니다.

import tensorflow as tf
### -> 딥러닝 모델 구축, 학습을 위해 사용하는 라이브러리입니다.
### -> 이 코드에서는 구체적으로 MAB 모델에서 사용자 데이터를 학습하고, 신경망 모델을 정의 및 훈련시킵니다.

from sklearn.neighbors import NearestNeighbors
### -> 음식 데이터에서 KNN을 사용해 유사한 데이터로 증강을 하기 위해 사용합니다.

from tensorflow.keras.models import Model
### -> MAB 모델에서 각 팔(Arm)에 대해 별도의 신경망 모델을 정의하기 위해 사용합니다.

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
### -> Keras라는 Tensorflow의 API를 통해 딥러닝 모델을 구성하는데 필요한 층(Layer)을 제공합니다.

from tensorflow.keras.optimizers import Adam
### -> 딥러닝 모델의 학습 속도를 조정하기 위한 최적화 알고리즘입니다. 뒤이어 후술할 Cold start Problem을 해결하기 위해 사용합니다.

from sklearn.preprocessing import StandardScaler
### -> 데이텨의 정규화를 위해 사용합니다. 평균을 0, 표준편차를 1로 변환합니다.

### **1. 데이터 로드 : 사용자 데이터 준비
def prepare_training_data(filepath):
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
                "liked": 1
            })
    return pd.DataFrame(data)

### -> 음식 데이터를 Excel 파일에서 읽어오며, 고유 ID를 추가합니다. 필요한 열만 필터링하여 반환합니다.

### **2. 데이터 증강
def hierarchical_knn_augment(user_data, food_data, k=5, target_size=1000):
    food_data.fillna(0, inplace=True)
    num_missing = target_size - len(user_data)
    if num_missing <= 0:
        return user_data

    food_data["Representative_Group"] = food_data["Representative Food Code"].astype(int)
    food_data["Subcategory_Group"] = food_data["Food Subcategory Code"].astype(int)
    feature_columns = ["Representative_Group", "Subcategory_Group", "calories", "protein", "fat", "carbs"]

    feature_array = food_data[feature_columns].to_numpy()
    knn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_model.fit(feature_array)

    new_data = []
    indices = np.random.choice(food_data.index, num_missing, replace=True)
    for idx in indices:
        query_point = feature_array[idx].reshape(1, -1)
        _, neighbors = knn_model.kneighbors(query_point)
        neighbor_indices = neighbors[0]
        mixed_data = food_data.iloc[neighbor_indices].mean(axis=0)
        mixed_data["liked"] = 1
        new_data.append(mixed_data)

    return pd.concat([user_data, pd.DataFrame(new_data)], ignore_index=True)

### -> KNN을 활용해 음식 데이터를 증강합니다. 기존 사용자 데이터가 부족한 경우, 유사한 음식 데이터를 기반으로 새로운 학습 데이터를 생성합니다.

### **3. Multi-Armed Bandit (MAB)
### **3-1. MAB 클래스 정의
class MultiArmedBandit:
    def __init__(self, food_data, arms, input_shape):
        self.arms = arms
        self.models = {arm: self.build_model(input_shape) for arm in arms}
        self.rewards = {arm: 0 for arm in arms}
        self.counts = {arm: 0 for arm in arms}
        self.scaler = StandardScaler()

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

### -> 각 음식 그룹(팔)을 학습하는 딥러닝 모델을 정의합니다. 각 팔은 별도의 모델을 사용하며, 보상을 업데이트하고 학습합니다.

### **3-2. 팔 선택 및 업데이트
    def choose_arm(self, epsilon=0.1):
        if np.random.rand() < epsilon:  # 탐험
            return np.random.choice(self.arms)
        else:  # 활용
            avg_rewards = {arm: self.rewards[arm] / (self.counts[arm] + 1e-6) for arm in self.arms}
            return max(avg_rewards, key=avg_rewards.get)

    def update(self, arm, reward):
        self.rewards[arm] += reward
        self.counts[arm] += 1

### -> Exploration(랜덤 선택)과 Exploitation(보상이 높은 팔 선택)을 통해 최적의 팔을 선택합니다. 이후 보상을 업데이트합니다.

### **4. 예측 및 추천
def predict_top_n(mab, test_features, top_n=3):
    predictions = []
    for arm in mab.arms:
        prediction = mab.predict(arm, test_features)
        predictions.append((arm, prediction[0][0]))
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    print(f"상위 {top_n}개의 팔과 예측 확률:")
    for i, (arm, prob) in enumerate(predictions[:top_n]):
        print(f"{i + 1}. 팔: {arm}, 좋아할 확률: {prob * 100:.2f}%")

### -> 테스트 데이터를 입력받아 각 음식 그룹에 대한 선호 확률을 계산하고, 가장 높은 확률을 가진 상위 N개의 결과를 출력합니다.

### **5. 실행 흐름
if __name__ == "__main__":
    user_data = prepare_training_data(USER_DATA_FILE)
    food_data = load_food_data(FOOD_DB_FILE)
    augmented_data = hierarchical_knn_augment(user_data, food_data, target_size=1000)

    arms = food_data["Representative Food Code"].unique()
    input_shape = (4,)
    mab = MultiArmedBandit(food_data, arms, input_shape)

    for step in range(100):
        arm = mab.choose_arm(epsilon=0.1)
        arm_data = augmented_data[augmented_data["Representative Food Code"] == arm]
        if len(arm_data) == 0:
            continue
        X = arm_data[["calories", "protein", "fat", "carbs"]].values
        y = arm_data["liked"].values
        reward = np.random.choice([0, 1], p=[1 - y.mean(), y.mean()])
        mab.train_arm(arm, X, y)
        mab.update(arm, reward)

    test_features = np.array([[500, 25, 10, 60]])  # 예시 데이터
    predict_top_n(mab, test_features, top_n=3)

### -> 데이터 로드, 증강, 학습, 테스트 데이터를 통한 예측까지의 전체 흐름을 실행합니다.
```

 3. 1번 데이터에 입각한 식단 구성 및 레시피 추천
  - 레시피 추천 시 중복 없고 비슷한 주재료 위주로 추천할 수 있도록
  - Item-based Recommendation Algorithm
   : 신규 사용자의 경우 cold-start problem이 발생하기에, Item-based Recommendation Algorithm을 사용해 사용자의 초기 입력값과 Dataset의 분류값을 기준으로 빠르게 사용자의 선호를 반영
 3. 추천해준 레시피 사용자 별로 저장
# Evaluation & Analysis
<img width="250" alt="image" src="https://github.com/user-attachments/assets/12e2c05d-1e9a-4071-a399-b4c3eb88cb9a">

# Related Work
 - AI+X:딥러닝 9주차 플립러닝 추천기술 영상
# Conclusion : Disscussion
