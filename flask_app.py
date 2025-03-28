from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
import requests

app = Flask(__name__)

# ✅ 1. 추천 알고리즘 실행 API
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        print("📌 [1] 추천 요청 수신됨")

        # === 파일 경로 설정 ===
        base_path = os.path.dirname(__file__)
        population_density_file = os.path.join(base_path, "data/데이터초안/34040_2023년_인구총괄(인구밀도).txt")
        child_safety_zone_file = os.path.join(base_path, "data/데이터초안/전국어린이보호구역표준데이터.csv")
        geo_mapping_file = os.path.join(base_path, "data/산출데이터/아산시_지역코드_좌표.csv")
        fire_station_file = os.path.join(base_path, "data/산출데이터/소방서_좌표_카카오.csv")
        result_output_file = os.path.join(base_path, "data/산출데이터/추천_수거함_위치.csv")

        print("📌 [2] 인구 밀도 데이터 로딩 중...")
        df_population = pd.read_csv(population_density_file, delimiter='^', header=None,
                                    names=['연도', '지역코드', '지표코드', '인구밀도'])
        df_population = df_population[df_population['지표코드'] == 'to_in_003']
        df_population_asan = df_population[df_population['지역코드'].astype(str).str.startswith("34040")].copy()

        print("📌 [3] 위경도 매핑 중...")
        df_geo = pd.read_csv(geo_mapping_file).dropna(subset=['위도', '경도'])
        df_population_asan = df_population_asan.merge(df_geo, on="지역코드", how="left")

        print("📌 [4] 고밀도 지역 필터링 중...")
        df_population_asan['인구밀도평균'] = df_population_asan['인구밀도'].rolling(window=5, center=True, min_periods=1).mean()
        df_population_asan['밀도차이'] = df_population_asan['인구밀도'] - df_population_asan['인구밀도평균']
        density_threshold = 0.8
        high_density_areas = df_population_asan[
            df_population_asan['밀도차이'] > df_population_asan['밀도차이'].quantile(density_threshold)].copy()

        min_recommendations = 20
        while len(high_density_areas) < min_recommendations and density_threshold > 0.5:
            density_threshold -= 0.05
            high_density_areas = df_population_asan[
                df_population_asan['밀도차이'] > df_population_asan['밀도차이'].quantile(density_threshold)].copy()

        print(f"✅ 고밀도 지역 후보 개수: {len(high_density_areas)}")

        print("📌 [5] 어린이 보호구역 필터링 중...")
        df_child_safety = pd.read_csv(child_safety_zone_file, encoding='cp949')[['소재지도로명주소', '경도', '위도']].dropna()
        safety_distance = 0.3
        safe_high_density_areas = []

        for _, pop_row in high_density_areas.iterrows():
            pop_loc = (pop_row['위도'], pop_row['경도'])
            if any(geodesic(pop_loc, (row['위도'], row['경도'])).km < safety_distance for _, row in df_child_safety.iterrows()):
                continue
            safe_high_density_areas.append(pop_row)

        df_safe = pd.DataFrame(safe_high_density_areas)
        print(f"✅ 어린이 보호구역 제외 후 지역 수: {len(df_safe)}")

        print("📌 [6] 소방서 거리 필터링 중...")
        df_fire = pd.read_csv(fire_station_file).dropna(subset=['위도', '경도'])
        fire_coords = df_fire[['위도', '경도']].values
        distances_to_fire = []
        for coord in df_safe[['위도', '경도']].values:
            fire_distances = [geodesic(coord, fire_coord).km for fire_coord in fire_coords]
            distances_to_fire.append(min(fire_distances))
        distances_to_fire = np.array(distances_to_fire)

        distance_threshold = np.percentile(distances_to_fire, 90)
        df_safe['is_far'] = distances_to_fire > distance_threshold
        df_safe = df_safe[df_safe['is_far'] == False]
        print(f"✅ 소방서 거리 제외 후 지역 수: {len(df_safe)}")

        print("📌 [7] DBSCAN 군집화 수행 중...")
        df_final_filtered = df_safe.copy()
        coords = np.radians(df_final_filtered[['위도', '경도']].values)
        epsilon_radians = (400 / 1000) / 6371
        dbscan = DBSCAN(eps=epsilon_radians, min_samples=5, metric='haversine')
        df_final_filtered['cluster'] = dbscan.fit_predict(coords)

        df_clustered = df_final_filtered[df_final_filtered['cluster'] != -1]
        df_noise = df_final_filtered[df_final_filtered['cluster'] == -1]
        cluster_centroids = df_clustered.groupby('cluster').agg({'위도': 'mean', '경도': 'mean'}).reset_index()
        final_recommendations = pd.concat([
            cluster_centroids[['위도', '경도']],
            df_noise[['위도', '경도']]
        ])

        print(f"✅ 군집화 완료 - 군집 수: {df_clustered['cluster'].nunique()}, 노이즈 수: {len(df_noise)}")

        print("📌 [8] 결과 CSV 저장 중...")
        final_recommendations.to_csv(result_output_file, index=False, encoding='utf-8-sig')
        print(f"🎯 추천 완료! 위치 수: {len(final_recommendations)}")
        print(f"💾 저장 경로: {result_output_file}")

        print("📌 [9] 시각화 이미지 저장 중...")
        plt.figure(figsize=(10, 6))
        plt.scatter(df_noise['경도'], df_noise['위도'], c='green', label='Non-clustered (Noise)', s=50)
        plt.scatter(df_clustered['경도'], df_clustered['위도'], c='blue', label='Clustered', s=50)
        plt.scatter(cluster_centroids['경도'], cluster_centroids['위도'], c='red', label='Cluster Centroids', s=100, marker='x')
        plt.legend()
        plt.title("Recommended Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "data/산출데이터/추천_수거함_시각화.png"))
        print("🖼️ 시각화 이미지 저장 완료")

        return jsonify({"message": "추천 알고리즘 실행 완료, 결과 저장됨 ✅"}), 200

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ 2. 기존 수거함과 비교하여 필터링된 추천 좌표 반환 API
@app.route('/recommend/compare', methods=['GET'])
def compare_existing_with_recommended():
    try:
        print("📌 [compare] 기존 수거함과 추천 위치 비교 시작")

        # === Spring Boot에서 기존 수거함 좌표 데이터 가져오기 ===
        spring_url = "http://localhost:8081/admin/findAllBox"
        response = requests.get(spring_url)
        response.raise_for_status()
        existing_boxes = response.json()
        print(f"📦 기존 수거함 수: {len(existing_boxes)}")

        # === 기존 수거함의 좌표 파싱 ===
        existing_coords = []
        for box in existing_boxes:
            location_str = box.get("location")
            if location_str and "POINT" in location_str:
                try:
                    lng, lat = map(float, location_str.replace("POINT (", "").replace(")", "").split())
                    existing_coords.append((lat, lng))  # 위도, 경도
                except ValueError as e:
                    print(f"⚠️ 위치 파싱 실패: {location_str}, 오류: {e}")

        # === 추천 위치 CSV 불러오기 ===
        base_path = os.path.dirname(__file__)
        recommended_file = os.path.join(base_path, "data/산출데이터/추천_수거함_위치.csv")
        df_recommended = pd.read_csv(recommended_file)
        print(f"📍 추천 위치 수 (CSV): {len(df_recommended)}")

        # === 비교: 기존 수거함과 100m 이내인 추천 위치 제거 ===
        filtered_recommendations = []
        for _, row in df_recommended.iterrows():
            rec_point = (row["위도"], row["경도"])
            too_close = False
            for exist_point in existing_coords:
                distance = geodesic(rec_point, exist_point).km
                if distance < 0.1:  # 100m 이내
                    too_close = True
                    break
            if not too_close:
                filtered_recommendations.append({"lat": rec_point[0], "lng": rec_point[1]})

        print(f"✅ 필터링 후 추천 위치 수: {len(filtered_recommendations)}")

        return jsonify(filtered_recommendations), 200

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)