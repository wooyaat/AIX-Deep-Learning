# AIX-Deep-Learning
# Title : 1인 가구를 위한 사용자 맞춤형 식단 추천 알고리즘
# Members 
  이호원, 전기공학과, howonlee1217@gmail.com / 박도연, 수학과, nsa09041@naver.com
# Proposal(Option A)
 - Motivation :
 - Goal : 
# Datasets
 - 식품의약품안전처 식품 영양성분 데이터베이스
 - https://various.foodsafetykorea.go.kr/nutrient/general/down/list.do
 - <img width="1469" alt="image" src="https://github.com/user-attachments/assets/d95f9afa-ea5b-4900-b091-1759f286c084">

# Methodology
 1. recomm.py 파일
 file_path = 'food_db2.xlsx'
 food_data = pd.read_excel(file_path)

 -> pandas 라이브러리를 활용해 엑셀 파일 데이터를 불러오는 함수입니다.
 
 if 'id' not in food_data.columns:
 food_data['id'] = food_data.index
 
 -> 불러온 엑셀의 음식 데이터에 id라 하는 고유 식별자 추가, 없다면 데이터 인덱스를 활용해 라벨링합니다.

 def get_user_input():
    gender = input("성별을 입력하세요 (male/female): ").strip().lower()
    age = int(input("나이를 입력하세요: "))
    weight = float(input("몸무게를 입력하세요 (kg): "))
    height = float(input("키를 입력하세요 (cm): "))
    activity_level = input("활동 수준을 선택하세요 (sedentary/lightly_active/moderately_active/very_active/super_active): ").strip()
    meal_count = int(input("하루 두 끼(2) 또는 세 끼(3) 중 선택하세요: "))
    return gender, age, weight, height, activity_level, meal_count
 
 -> 아래 정의할 calculate_daily_calories 함수를 계산하기 위해 사용자로부터 정보를 입력받는 함수입니다.

 def calculate_daily_calories(gender, age, weight, height, activity_level):
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
 
 -> 입력 받은 사용자의 정보를 기반으로 기초 대사량을 계산합니다.

     activity_multiplier = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "super_active": 1.9,
    }
    return bmr * activity_multiplier.get(activity_level, 1.2)
 
  -> 사용자에게 활동 수준을 입력 받아 성별을 기반으로 계산한 기초대사량에 활동 계수를 곱합니다. 

 def recommend_meals(food_data, meal_calories, exclude_ids, number=3):
    filtered_data = food_data[
        (food_data['calories'] <= meal_calories) & 
        (~food_data['id'].isin(exclude_ids))
    ].sort_values(by='calories', ascending=False).head(number)
    return filtered_data

-> 위에서 얻어낸 사용자 정보를 토대로 조건 필터링을 통해 메뉴를 추천하는 함수입니다. 끼니 당 권장 칼로리를 넘지 않는 음식을 기준으로 추천을 진행하며, 여러 끼니를 추천해야 하는 특성을 고려해 중복되지 않도록 추천을 진행합니다.

 def save_user_data(user_id, input_data, recommendations):
    user_data_file = "user_data.json"
    if os.path.exists(user_data_file):
        try:
            with open(user_data_file, "r") as f:
                user_data = json.load(f)
        except json.JSONDecodeError:
            user_data = {}
    else:
        user_data = {}

-> 사용자에게 추천한 메뉴를 저장하기 위한 함수입니다. 데이터는 JSON 형태입니다.

    recommendations_dict = recommendations.astype(str).to_dict(orient="records")
    user_data[user_id] = {
        "input_data": input_data,
        "recommendations": recommendations_dict
    }
    with open(user_data_file, "w") as f:
        json.dump(user_data, f, indent=4)
    print("사용자 데이터가 저장되었습니다.")

-> JSON 파일에 저장한 데이터를 리스트 형태로 

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
