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
# Methodology
 1. 사용자 입력값에 따른 하루 섭취 칼로리 및 영양 성분 계산
 2. 1번 데이터에 입각한 식단 구성 및 레시피 추천
  - 레시피 추천 시 중복 없고 비슷한 주재료 위주로 추천할 수 있도록
  - Item-based Recommendation Algorithm
   : 신규 사용자의 경우 cold-start problem이 발생하기에, Item-based Recommendation Algorithm을 사용해 사용자의 초기 입력값과 Dataset의 분류값을 기준으로 빠르게 사용자의 선호를 반영
 3. 추천해준 레시피 사용자 별로 저장
# Evaluation & Analysis
<img width="250" alt="image" src="https://github.com/user-attachments/assets/12e2c05d-1e9a-4071-a399-b4c3eb88cb9a">

# Related Work
 - AI+X:딥러닝 9주차 플립러닝 추천기술 영상
# Conclusion : Disscussion
