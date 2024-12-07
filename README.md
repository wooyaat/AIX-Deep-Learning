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
  데이터 로드
python
코드 복사
file_path = 'food_db2.xlsx'
food_data = pd.read_excel(file_path)
-> pandas 라이브러리를 활용하여 Excel 파일(food_db2.xlsx)에서 음식 데이터를 불러옵니다. 이 데이터는 음식의 영양 성분과 같은 정보를 포함합니다.

python
코드 복사
if 'id' not in food_data.columns:
    food_data['id'] = food_data.index
-> 불러온 음식 데이터에 고유 식별자인 id 열을 추가합니다. 만약 id 열이 없다면, 데이터의 인덱스를 활용하여 각 항목에 고유 번호를 부여합니다.

사용자 입력 처리
python
코드 복사
def get_user_input():
    gender = input("성별을 입력하세요 (male/female): ").strip().lower()
    age = int(input("나이를 입력하세요: "))
    weight = float(input("몸무게를 입력하세요 (kg): "))
    height = float(input("키를 입력하세요 (cm): "))
    activity_level = input("활동 수준을 선택하세요 (sedentary/lightly_active/moderately_active/very_active/super_active): ").strip()
    meal_count = int(input("하루 두 끼(2) 또는 세 끼(3) 중 선택하세요: "))
    return gender, age, weight, height, activity_level, meal_count
-> 사용자에게 성별, 나이, 몸무게, 키, 활동 수준, 하루 식사 횟수를 입력받는 함수입니다. 이 정보는 이후 칼로리 계산과 식단 추천에 사용됩니다.

하루 권장 칼로리 계산
python
코드 복사
def calculate_daily_calories(gender, age, weight, height, activity_level):
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
-> 사용자 입력 정보를 기반으로 **기초 대사량(BMR)**을 계산합니다. 남성과 여성의 계산식이 다릅니다.

python
코드 복사
    activity_multiplier = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "super_active": 1.9,
    }
    return bmr * activity_multiplier.get(activity_level, 1.2)
-> 기초 대사량(BMR)에 사용자의 활동 수준에 따른 계수를 곱하여 하루 권장 칼로리(TDEE)를 계산합니다.

식단 추천
python
코드 복사
def recommend_meals(food_data, meal_calories, exclude_ids, number=3):
    filtered_data = food_data[
        (food_data['calories'] <= meal_calories) &
        (~food_data['id'].isin(exclude_ids))
    ].sort_values(by='calories', ascending=False).head(number)
    return filtered_data
-> 사용자의 끼니당 권장 칼로리 범위 안에서 음식 데이터를 필터링하고, 칼로리 기준으로 정렬하여 상위 number개의 음식을 추천합니다.
-> 이미 선택한 음식(exclude_ids)은 추천 대상에서 제외합니다.

추천 결과 저장
python
코드 복사
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
-> 기존에 저장된 user_data.json 파일을 읽어옵니다. 파일이 없거나 손상된 경우 초기화합니다.

python
코드 복사
    recommendations_dict = recommendations.astype(str).to_dict(orient="records")

    user_data[user_id] = {
        "input_data": input_data,
        "recommendations": recommendations_dict
    }

    with open(user_data_file, "w") as f:
        json.dump(user_data, f, indent=4)
    print("사용자 데이터가 저장되었습니다.")
-> 사용자의 입력 정보와 추천된 음식을 JSON 파일에 저장합니다.
-> to_dict 메서드를 사용하여 DataFrame을 JSON에 저장 가능한 형식으로 변환합니다.

사용자 선택 처리
python
코드 복사
def get_user_choice(recommendations):
    print("\n추천된 메뉴:")
    for idx, row in recommendations.iterrows():
        print(f"{row['id']}: {row['name']} - {row['calories']} kcal")

    selected_id = int(input("선호하는 메뉴의 ID를 선택하세요: "))
    selected_menu = recommendations[recommendations['id'] == selected_id]
    return selected_menu
-> 추천된 메뉴를 출력하고, 사용자로부터 선택한 메뉴의 ID를 입력받아 반환하는 함수입니다.

메인 실행 흐름
python
코드 복사
if __name__ == "__main__":
    gender, age, weight, height, activity_level, meal_count = get_user_input()
    daily_calories = calculate_daily_calories(gender, age, weight, height, activity_level)
    meal_calories = daily_calories / meal_count
-> 사용자 입력을 처리하고, 하루 권장 칼로리 및 끼니당 권장 칼로리를 계산합니다.

python
코드 복사
    for meal_type in ["meal_1", "meal_2", "meal_3"][:meal_count]:
        recommendations = recommend_meals(food_data, meal_calories, exclude_ids)
        if not recommendations.empty:
            selected_menu = get_user_choice(recommendations)
            all_recommendations.append(selected_menu)
            exclude_ids.extend(selected_menu['id'].tolist())
-> 끼니마다 음식을 추천하고, 사용자가 선택한 메뉴를 제외 목록에 추가합니다.

python
코드 복사
    if all_recommendations:
        save_user_data(user_id, input_data, pd.concat(all_recommendations))
    else:
        print("추천할 메뉴가 없습니다.")
-> 최종적으로 추천 결과를 JSON 파일에 저장합니다. 추천할 음식이 없는 경우 메시지를 출력합니다.



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
