import pandas as pd
import json
import os

# 데이터 로드
file_path = 'food_db2.xlsx'  # 파일 경로
food_data = pd.read_excel(file_path)

# id 열 추가 (존재하지 않을 경우)
if 'id' not in food_data.columns:
    food_data['id'] = food_data.index  # 고유 ID 추가

# 사용자 입력 함수
def get_user_input():
    gender = input("성별을 입력하세요 (male/female): ").strip().lower()
    age = int(input("나이를 입력하세요: "))
    weight = float(input("몸무게를 입력하세요 (kg): "))
    height = float(input("키를 입력하세요 (cm): "))
    activity_level = input("활동 수준을 선택하세요 (sedentary/lightly_active/moderately_active/very_active/super_active): ").strip()
    meal_count = int(input("하루 두 끼(2) 또는 세 끼(3) 중 선택하세요: "))
    return gender, age, weight, height, activity_level, meal_count

# 하루 권장 칼로리 계산 함수
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

# 식단 추천 함수
def recommend_meals(food_data, meal_calories, exclude_ids, number=3):
    """
    조건에 맞는 메뉴를 추천합니다.
    :param food_data: DataFrame, 음식 데이터베이스
    :param meal_calories: 끼니당 권장 칼로리
    :param exclude_ids: 제외할 메뉴 ID 리스트
    :param number: 추천할 메뉴 개수
    """
    filtered_data = food_data[
        (food_data['calories'] <= meal_calories) &
        (~food_data['id'].isin(exclude_ids))  # 제외된 ID 필터링
    ].sort_values(by='calories', ascending=False).head(number)
    
    return filtered_data

# 사용자 데이터 저장 함수
def save_user_data(user_id, input_data, recommendations):
    """
    사용자 입력 및 추천 결과를 JSON 파일에 저장
    :param user_id: 사용자 고유 ID
    :param input_data: 사용자 입력 데이터 (딕셔너리)
    :param recommendations: 추천된 메뉴 리스트 (데이터프레임)
    """
    user_data_file = "user_data.json"
    if os.path.exists(user_data_file):
        try:
            with open(user_data_file, "r") as f:
                user_data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSON 파일이 비어 있거나 손상되었습니다. 초기화합니다.")
            user_data = {}
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

# 사용자 선택 처리 함수
def get_user_choice(recommendations):
    """
    추천된 메뉴에서 사용자의 선택을 받습니다.
    :param recommendations: 추천된 메뉴 리스트 (DataFrame)
    :return: 선택된 메뉴 (DataFrame)
    """
    print("\n추천된 메뉴:")
    for idx, row in recommendations.iterrows():
        print(f"{row['id']}: {row['name']} - {row['calories']} kcal")

    selected_id = int(input("선호하는 메뉴의 ID를 선택하세요: "))
    selected_menu = recommendations[recommendations['id'] == selected_id]
    return selected_menu

# 메인 실행 코드
if __name__ == "__main__":
    gender, age, weight, height, activity_level, meal_count = get_user_input()

    daily_calories = calculate_daily_calories(gender, age, weight, height, activity_level)
    print(f"하루 권장 칼로리: {daily_calories:.2f} kcal")

    meal_calories = daily_calories / meal_count
    print(f"끼니당 권장 칼로리: {meal_calories:.2f} kcal")

    input_data = {
        "gender": gender,
        "age": age,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "meal_count": meal_count,
        "daily_calories": daily_calories,
        "meal_calories": meal_calories,
    }

    user_id = "user_1"
    all_recommendations = []
    exclude_ids = []

    for meal_type in ["meal_1", "meal_2", "meal_3"][:meal_count]:
        print(f"\n[{meal_type.capitalize()} 메뉴 추천]")
        recommendations = recommend_meals(food_data, meal_calories, exclude_ids)
        if not recommendations.empty:
            selected_menu = get_user_choice(recommendations)  # 사용자 선택
            all_recommendations.append(selected_menu)
            exclude_ids.extend(selected_menu['id'].tolist())  # 선택된 메뉴를 제외 목록에 추가

    if all_recommendations:
        save_user_data(user_id, input_data, pd.concat(all_recommendations))
    else:
        print("추천할 메뉴가 없습니다.")
