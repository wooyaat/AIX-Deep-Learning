# AIX-Deep-Learning
# Title : 1인 가구를 위한 사용자 맞춤형 식단 추천 알고리즘
# Members 
  이호원, 전기공학과, howonlee1217@gmail.com / 박도연, 수학과, nsa09041@naver.com
# Proposal(Option A)
 - Motivation : 자취생의 식사 메뉴 고민을 해결하기 위해 영양학적으로 균형 잡혀있으면서 다양한 식단을 추천해주고자 함.
 - Goal : 사용자 정보를 기반으로 하루 권장 칼로리를 계산 후 추천 식단을 제공, 데이터가 적을 때 데이터 증강을 활용해 딥러닝 모델 구현하기
# Datasets
 - 식품의약품안전처 식품 영양성분 데이터베이스
 - https://various.foodsafetykorea.go.kr/nutrient/general/down/list.do
 - <img width="1469" alt="image" src="https://github.com/user-attachments/assets/d95f9afa-ea5b-4900-b091-1759f286c084">

# Methodology_Concept

# Cold-Start Problem
: 추천 기술에서 해결해야 할 문제로, 새로운 유저가 서비스를 처음 사용할 때 그 유저에 대해 충분한 정보가 수집되지 못한 상태에서 적절한 제품을 추천하지 못하는 문제입니다. 이 프로젝트에서는 Cold-Start Problem을 해결하기 위해 KNN 기반 데이터 증강과 Multi-Armed Bandit System을 활용할 예정입니다.

# 데이터 증강
: 가존 데이터를 변형하거나 새로운 데이터를 생성해서 학습 데이터를 확장하는 방법입니다. 신규 사용자에 대한 정보가 적은 상황에서 딥러닝 모델의 예측 정확도를 향상시키기 위해 시행합니다.

# KNN
: K-Nearest Neighbors의 약자로, 유클리드 거리와 가장 가까운 K개의 데이터라는 두 가지 hyperparameter를 사용해 데이터를 분류하는 지도학습 알고리즘의 일종입니다.

# Multi-Armed Bandit System
![image](https://github.com/user-attachments/assets/4b30719c-4568-46d4-9080-51c1279bc8ca)
 
 : 이 시스템은 탐색(Exploration)과 활용(Exploitation)을 모두 적절하게 활용해 보상을 최대화햔다는 목표를 수행하는 시스템입니다. 이 시스템은 서로 다른 보상 확률을 가지는 팔(arm)들, 매 시간 단꼐에서 하나의 팔을 선택하는 행동(action), 행동을 통해 선택한 팔에서 얻어낸 결과인 보상 (reward), 누적 보상을 최대화해내는 목표(goal)로 구성되어 있습니다. 탐색은 정보에 기반하지 않고 팔을 시험적으로 선택해 정보를 얻는 과정으로, 새로운 선택지를 파악할 수 있으나 데이터에 기반한 보상을 선택하지 못할 수 있다는 단점이 있습니다. 활용은 현재까지의 데이터를 기반으로 가장 보상이 높응 것이라 예상되는 팔을 선택해 정보를 얻는 과정으로, 행동을 실행하는 시점에서 가장 높은 보상을 얻을 수 있으나 최적의 팔을 찾지 못할 위험이 있습니다.
   
 : 수학적으로 행동의 근거는 다양한 알고리즘이 있지만, 이 코드에서는 Epsilon-Greedy 알고리즘을 활용합니다. 이 알고리즘은 탐색과 활용의 빈도를 각각 ϵ과 1-ϵ의 확률로 진행하며, 간단하고 구현이 쉽다는 장점이 있습니다. Upper Confidence Bound나 Thompson Sampling 알고리즘을 사용하지 않은 이유는 구련 과정에서 베이즈 추론 지식이 요구되기에 Epsilon-Greedy 알고리즘보다 설명이 어렵기 때문입니다.

# Methodology_Apply
앞서 설명한 내용들을 코드에 활용한 방법에 대해 설명하겠습니다. Cold-Start Problem을 해결하기 위해 데이터 증강, KNN, MAB 시스템을 활용했는데, 우선 처음 사용자가 서비스를 사용할 때 신체 정보를 입력하며 간단하게 한 번 식단을 추천하도록 인공지능 없는 파이썬 코드를 작성했습니다. 그 정보를 토대로 KNN 기반 데이터 증강을 진행합니다. 

증강 과정은 다음과 같습니다. food_bd2.xslx 데이터에 존재하는 Representative Food Code와 Food SubCategory Code 데이터를 활용해    계층적 그룹화를 진행하고 추가적으로 칼로리, 탄수화물, 단백질, 지방 데이터까지 고려해 특성 배열을 생성합니다. 특성 배열 데이터를 활용해 KNN 햑습을  진행합니다. 이후 랜덤하게 데이터를 생성하는데, 이 과정에서 KNN의 hyperparameter K를 활용해 이웃 데이터룰 찾고 그 데이터들의 평균값으로 새로운  데이터를 생성합니다. 이후 MAB 시스템을 통해 추천을 진행합니다.

이 적용 과정의 특징은 KNN 기반 데이터 증강입니다. 데이터의 분류 코드가 존재했기에 이를 활용해 초기 사용자의 선택과 유사한 데이터를 확보할 수 있었고, 따라서 딥러닝 구현이 가능할 정도로 데이터를 확보하면서도 그 데이터의 정확성을 유지할 수 있습니다.

# Methodology_Code
# recomm.py
1. **칼로리 계산**: 사용자의 기본 정보를 바탕으로 하루 권장 칼로리를 계산합니다.
2. **식단 추천**: 권장 칼로리 범위 내에서 적합한 음식을 추천합니다.
3. **데이터 저장**: 사용자 입력 및 추천 결과를 JSON 파일에 저장합니다.
---
## **코드 주요 부분**

```python
### **0. 코드 실행에 필요한 라이브러리와 모듈 준비
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

# recomm_ㅡmab2.py
1. **데이터 로드 및 전처리**: 음식 데이터와 사용자 데이터를 읽어오고 학습 가능한 형태로 준비합니다.
2. **데이터 증강**: 권장 칼로리 범위 내에서 적합한 음식을 추천합니다.
3. **Multi-Armed Bandit(MAB)**: 각 음식 그룹(팔)을 학습하고 보상 기반으로 최적의 음식을 선택합니다.
4. **추천 출력**: 사용자가 선호할 가능성이 높은 음식 상위 3개를 예측하여 제공합니다.
---
## **코드 주요 부분**

```python
### **0. 코드 실행에 필요한 라이브러리와 모듈 준비
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

# Evaluation & Analysis
<img width="250" alt="image" src="https://github.com/user-attachments/assets/12e2c05d-1e9a-4071-a399-b4c3eb88cb9a">
Representative Food Code

# Related Work
 - AI+X:딥러닝 9주차 플립러닝 SKT AI CURRICULUM 추천기술
   : 해당 영상에서 추천 기술에서 발생할 수 있는 Cold Start Problem과 Multi-Armed bandit system 아이디어를 얻어 적용할 수 있었습니다.
 - https://techblog-history-younghunjo1.tistory.com/166#google_vignette
   : Cold-Start Problem의 정의에 대해 참고했습니다.
 - https://velog.io/@jhlee508/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-KNNK-Nearest-Neighbor-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
   : KNN 알고리즘에 대해 참고했습니다.
 - https://myeonghak.github.io/recommender%20systems/RecSys-Multi-Armed-Bandits(MAB)/
   :MAB와 관련된 내용을 참고했습니다.

# Conclusion : Disscussion
 "그거 인공지능으로 하면 되는 거 아니야?" 라는 말을 정말 여러 사람들, 여러 업계에서 들을 수 있는 나날들을 보내고 있다 생각합니다. 이 아이디어도 단순하게 그 질문에서 출발했습니다. 자취생 동기들끼리 수업 후 함께 집에 가며 저녁 메뉴를 고민하던 그 순간, 인공지능으로 문제를 해결해보자는 생각에서 이 프로젝트는 시작되었습니다.
 
 하지만 현실은 녹록지 않았고, 질문 한 마디 정도로 단순하지 않았습니다. 프로젝트를 진행하며 겪었던 큰 문제들은 다음과 같았습니다.
 1. 데이터 찾기
  : 사실 처음 주제는 1인 가구를 위한 식단 및 레시피 추천으로, 구현해낸 기능에 추천한 음식의 레시피 정보 역시 제공하는 것이 목적이었습니다. 하지만  
    레시피와 식단, 그에 따른 영양성분까지 모두 포함된 데이터를 찾는데 어려움을 겪었고, 초기엔 SPOONACULAR라는 API를 사용하려 했으나 한국에    
    어울리지 않는 식단들과 금전적인 문제로 레시피 추천과 관련된 기능을 구현해낼 수 없었습니다.
    하지만 데이터와 관련해 항상 어려운 점만 있었던 것은 아니었습니다. 새롭게 찾은 현 데이터, 식약처에서 제공하는 식품영양성분 데이터베이스의 경우   
    음식의 대분류, 중분류, 소분류를 코드로 진행했기 때문에 데이터 증강 과정에서 KNN 사용이 가능했습니다. 이는 증강의 인과성을 확보해 사용자화 증강 
    데이터를 확보할 수 있었고, MAB 모델의 정확도를 높이는데 기여했습니다.

 2. Cold Start Problem
  : 이론적인 인공지능 학습과 실전 서비스 구축의 괴리를 가장 크게 느꼈던 대표적 사례였습니다. 결국 인공지능 기술을 서비스로 만들어 사용자에게 전달하 
    기 위해선 사용자의 입장에서 사소한 문제들에 대해 고민해보고 이를 기술적으로 반영해야 하는데, 그 과정에서 가장 어려웠던 요소가 바로 Cold start 
    problem이었습니다. 추천 알고리즘에서는 초기 진입 사용자의 데이터가 많지 않기 때문에 원활한 추천을 진행하기 어려워 데이터가 적은 상황에서도    
    추천의 정확도를 증가시킬 수 있는 방법이 필요합니다. 이 프로젝트에서는 해당 문제를 해결하기 위해 데이터의 분류코드를 활용한 KNN을 통해 데이터   
    증강을 적용했고, MAB 모델이 보다 정확하게 학습하여 추천을 진행할 수 있었습니다.

 
