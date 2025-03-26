import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import requests
import matplotlib.pyplot as plt  # 시각화를 위해 추가

# === 파일 경로 ===
population_density_file = "../data/데이터초안/34040_2023년_인구총괄(인구밀도).txt"
child_safety_zone_file = "../data/데이터초안/전국어린이보호구역표준데이터.csv"
geo_mapping_file = "../data/산출데이터/아산시_지역코드_좌표.csv"
fire_station_file = "../data/산출데이터/소방서_좌표_카카오.csv"  # 위도, 경도 포함된 소방서 데이터

# === 기존 수거함 위치 불러오기 ===
try:
    response = requests.get("http://localhost:8081/admin/findAllBox")
    response.raise_for_status()
    existing_boxes = pd.DataFrame(response.json())

    # 'location' 컬럼에서 경도와 위도를 분리하여 새로운 '위도'와 '경도' 컬럼 생성
    existing_boxes[['경도', '위도']] = existing_boxes['location'].str.extract(r'POINT \(([-\d.]+) ([-\d.]+)\)').astype(float)

    print("✅ 기존 수거함 데이터 불러오기 성공")
except Exception as e:
    raise RuntimeError("❌ 기존 수거함 API 요청 실패: ", e)

# === 인구 밀도 데이터 로드 ===
df_population = pd.read_csv(population_density_file, delimiter='^', header=None,
                            names=['연도', '지역코드', '지표코드', '인구밀도'])
df_population = df_population[df_population['지표코드'] == 'to_in_003']
df_population_asan = df_population[df_population['지역코드'].astype(str).str.startswith("34040")].copy()

# === 위경도 매핑 ===
df_geo = pd.read_csv(geo_mapping_file).dropna(subset=['위도', '경도'])
df_population_asan = df_population_asan.merge(df_geo, on="지역코드", how="left")

# === 밀도 차이 기반 고밀도 지역 추출 ===
df_population_asan['인구밀도평균'] = df_population_asan['인구밀도'].rolling(window=5, center=True, min_periods=1).mean()
df_population_asan['밀도차이'] = df_population_asan['인구밀도'] - df_population_asan['인구밀도평균']
density_threshold = 0.8  # 밀도 차이 기준
high_density_areas = df_population_asan[
    df_population_asan['밀도차이'] > df_population_asan['밀도차이'].quantile(density_threshold)].copy()

min_recommendations = 20
while len(high_density_areas) < min_recommendations and density_threshold > 0.5:
    density_threshold -= 0.05
    high_density_areas = df_population_asan[
        df_population_asan['밀도차이'] > df_population_asan['밀도차이'].quantile(density_threshold)].copy()

# === 어린이 보호구역 데이터 필터 ===
df_child_safety = pd.read_csv(child_safety_zone_file, encoding='cp949')[['소재지도로명주소', '경도', '위도']].dropna()
safety_distance = 0.3  # 300m
safe_high_density_areas = []

for _, pop_row in high_density_areas.iterrows():
    pop_loc = (pop_row['위도'], pop_row['경도'])
    if any(geodesic(pop_loc, (row['위도'], row['경도'])).km < safety_distance for _, row in df_child_safety.iterrows()):
        continue
    safe_high_density_areas.append(pop_row)

df_safe = pd.DataFrame(safe_high_density_areas)

# === 소방서 데이터 불러오기 ===
df_fire = pd.read_csv(fire_station_file).dropna(subset=['위도', '경도'])

# === 소방서와 너무 멀리 떨어진 지역 제외 ===
fire_coords = df_fire[['위도', '경도']].values
distances_to_fire = []
for coord in df_safe[['위도', '경도']].values:
    fire_distances = [geodesic(coord, fire_coord).km for fire_coord in fire_coords]
    distances_to_fire.append(min(fire_distances))  # 가장 가까운 소방서와의 거리

distances_to_fire = np.array(distances_to_fire)

# 상위 10%가 너무 멀리 떨어진 위치로 판단하여 제외
distance_threshold = np.percentile(distances_to_fire, 90)
df_safe['is_far'] = distances_to_fire > distance_threshold

df_safe = df_safe[df_safe['is_far'] == False]

# === 기존 수거함과 가까운 위치 제외 ===
box_distance_limit = 0.3  # 300m
final_filtered = []

for _, row in df_safe.iterrows():
    candidate = (row['위도'], row['경도'])
    # 기존 수거함과의 거리 계산 후, 300m 이내에 있으면 제외
    if all(geodesic(candidate, (box['위도'], box['경도'])).km >= box_distance_limit for _, box in existing_boxes.iterrows()):
        final_filtered.append(row)

# === 데이터 분포 확인 ===
df_final_filtered = pd.DataFrame(final_filtered)
print(f"총 위치 개수: {len(df_final_filtered)}")

# 위치 간 거리 분포 확인
distances = []
coords_list = df_final_filtered[['위도', '경도']].values
for i in range(len(coords_list)):
    for j in range(i + 1, len(coords_list)):
        dist = geodesic(coords_list[i], coords_list[j]).km
        distances.append(dist)

distances = np.array(distances)
print(f"위치 간 거리 통계 (km):")
print(f"최소 거리: {distances.min():.3f}")
print(f"최대 거리: {distances.max():.3f}")
print(f"평균 거리: {distances.mean():.3f}")
print(f"중앙값 거리: {np.median(distances):.3f}")

# === DBSCAN 군집화 ===
coords = np.radians(df_final_filtered[['위도', '경도']].values)

# 미터 단위를 라디안으로 변환
EARTH_RADIUS_KM = 6371  # 지구 반지름 (km)
epsilon_meters = 400  # 400m로 설정
epsilon_radians = (epsilon_meters / 1000) / EARTH_RADIUS_KM  # 미터 -> km -> 라디안
min_samples = 5

print(f"epsilon (meters): {epsilon_meters}")
print(f"epsilon (radians): {epsilon_radians}")

dbscan = DBSCAN(eps=epsilon_radians, min_samples=min_samples, metric='haversine')
df_final_filtered['cluster'] = dbscan.fit_predict(coords)

# 군집화 결과 확인
cluster_counts = df_final_filtered['cluster'].value_counts()
print("군집화된 클러스터 개수:")
print(cluster_counts)

# 군집화된 지역과 노이즈 분리
df_final_filtered_clustered = df_final_filtered[df_final_filtered['cluster'] != -1]
df_final_filtered_noise = df_final_filtered[df_final_filtered['cluster'] == -1]

# 군집화된 지역의 중심 계산 (빨간색 점)
cluster_centroids = df_final_filtered_clustered.groupby('cluster').agg({'위도': 'mean', '경도': 'mean'}).reset_index()

# 최종 추천 위치: 군집 중심(빨간색) + 군집화되지 않은 위치(초록색)
final_recommendations = pd.concat([cluster_centroids[['위도', '경도']], df_final_filtered_noise[['위도', '경도']]])

# === 결과 출력 ===
print("🎯 최종 추천 위치 (기존 수거함과 300m 이상 거리 제외):")
print(final_recommendations)

# 결과를 파일로 저장
final_recommendations.to_csv("data/산출데이터/추천_수거함_위치.csv", index=False, encoding='utf-8-sig')
print(f"\n💾 추천 좌표가 '추천_수거함_위치.csv'에 저장되었습니다.")

# === Spring Boot로 데이터 전송 ===
# CSV 파일 읽기
recommended_locations = pd.read_csv("../data/산출데이터/추천_수거함_위치.csv")

# 데이터프레임을 JSON 형식으로 변환
# '위도'와 '경도'를 'location' 형식으로 변환 (예: "POINT (경도 위도)")
recommended_locations['location'] = recommended_locations.apply(
    lambda row: f"POINT ({row['경도']} {row['위도']})", axis=1
)
# 필요한 컬럼만 선택 (Spring Boot에서 기대하는 형식에 맞게)
data_to_send = recommended_locations[['location']].to_dict('records')

# Spring Boot 서버로 POST 요청 전송
url = "http://localhost:8081/admin/saveRecommendedBoxes"  # Spring Boot 엔드포인트
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=data_to_send, headers=headers)
    response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
    print("✅ Spring Boot로 추천 위치 전송 성공")
    print(f"서버 응답: {response.text}")
except Exception as e:
    print(f"❌ Spring Boot로 전송 실패: {e}")

# === 시각화 ===
plt.figure(figsize=(10, 6))
# 노이즈 (초록색)
plt.scatter(df_final_filtered_noise['경도'], df_final_filtered_noise['위도'], c='green', label='Non-clustered (Noise)', s=50)
# 군집화된 위치 (파란색)
plt.scatter(df_final_filtered_clustered['경도'], df_final_filtered_clustered['위도'], c='blue', label='Clustered', s=50)
# 군집 중심 (빨간색)
plt.scatter(cluster_centroids['경도'], cluster_centroids['위도'], c='red', label='Cluster Centroids', s=100, marker='x')
plt.legend()
plt.title("Recommended Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()