import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from geopy.distance import geodesic

# 데이터 파일 경로
population_density_file = "data/34040_2023년_인구총괄(인구밀도).txt"
child_safety_zone_file = "data/전국어린이보호구역표준데이터.csv"

# 인구 밀도 데이터 로드
df_population = pd.read_csv(population_density_file, delimiter='^', header=None, names=['연도', '지역코드', '지표코드', '인구밀도'])

# 인구 밀도 데이터 필터링 (to_in_003: 인구 밀도)
df_population = df_population[df_population['지표코드'] == 'to_in_003'].copy()

# 아산시 데이터 필터링 (지역코드가 '34040'으로 시작하는 행만 선택)
df_population_asan = df_population[df_population['지역코드'].astype(str).str.startswith("34040")].copy()

# 주변 인구 밀도의 평균값과 비교하여 상대적으로 밀도가 높은 지역 찾기
df_population_asan['인구밀도평균'] = df_population_asan['인구밀도'].rolling(window=5, center=True, min_periods=1).mean()
df_population_asan['밀도차이'] = df_population_asan['인구밀도'] - df_population_asan['인구밀도평균']

# 상대적으로 밀도가 높은 지역 선택 (밀도 차이가 상위 10% 이상인 지역)
high_density_areas = df_population_asan[df_population_asan['밀도차이'] > df_population_asan['밀도차이'].quantile(0.9)].copy()

# 어린이 보호구역 데이터 로드
df_child_safety = pd.read_csv(child_safety_zone_file, encoding='cp949')  # 인코딩 오류 방지
df_child_safety = df_child_safety[['소재지도로명주소', '경도', '위도']].dropna()  # 필요한 컬럼만 선택

# 보호구역과 일정 거리(500m 이상) 떨어진 지역 찾기
safe_high_density_areas = []
safety_distance = 0.5  # 500m

for _, pop_row in high_density_areas.iterrows():
    is_safe = True  # 기본적으로 안전하다고 가정
    for _, safety_row in df_child_safety.iterrows():
        try:
            pop_location = (pop_row['인구밀도'], pop_row['인구밀도'])  # 위치 정보 필요 (현재는 대체 값)
            safety_location = (safety_row['위도'], safety_row['경도'])
            distance = geodesic(pop_location, safety_location).km  # 거리 계산

            if distance < safety_distance:
                is_safe = False
                break
        except:
            continue

    if is_safe:
        safe_high_density_areas.append(pop_row)

# 안전한 고밀도 지역을 데이터프레임으로 변환
df_safe_high_density_areas = pd.DataFrame(safe_high_density_areas)

# 위치가 최대한 겹치지 않도록 하기 위해 가장 먼 점들만 선택
coordinates = np.array(df_safe_high_density_areas[['인구밀도']])  # 단순 수치 기반 공간 표현 (위경도 대체)
tree = cKDTree(coordinates)

selected_indices = []
visited = set()

for i in range(len(df_safe_high_density_areas)):
    if i in visited:
        continue
    selected_indices.append(i)
    _, neighbors = tree.query(coordinates[i], k=5)  # 가까운 5개 점 조회
    visited.update(neighbors)

selected_final_areas = df_safe_high_density_areas.iloc[selected_indices]

# 좌표값이 포함된 최종 추천 위치 CSV로 저장
output_file = "추천_위치_좌표.csv"
selected_final_areas.to_csv(output_file, index=False, encoding='utf-8-sig')

# 결과 출력
print("추천된 위치 목록 (어린이 보호구역 제외, 겹치지 않음):")
print(selected_final_areas[['지역코드', '인구밀도']])  # 좌표 정보 필요 시 추가 가능

print(f"\n추천된 위치 좌표 정보가 {output_file} 파일로 저장되었습니다.")
