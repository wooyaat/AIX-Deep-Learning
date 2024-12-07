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

### **1. 데이터 로드**
```python
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
