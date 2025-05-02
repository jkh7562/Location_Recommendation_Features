from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
import requests
import time
import geopandas as gpd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 로드
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
BACKEND_ORIGIN = os.getenv("BACKEND_ORIGIN")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8081"}})


# 📌 주소 → 좌표 변환 함수
def kakao_geocode(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        result = response.json()
        if result["documents"]:
            first = result["documents"][0]
            return float(first["y"]), float(first["x"])  # 위도, 경도
        else:
            return None, None
    except Exception as e:
        print(f"❌ 주소 변환 실패: {address} / {e}")
        return None, None


# ✅ 소방서 및 어린이보호구역 좌표만 반환하는 API
@app.route('/get-coordinates', methods=['GET'])
def get_coordinates():
    try:
        print("📌 소방서 및 어린이보호구역 좌표 요청 수신")

        # 데이터 경로 설정
        base_path = os.path.dirname(__file__)
        fire_station_file = os.path.join(base_path, "data/산출데이터/소방서_좌표_카카오.csv")
        child_safety_file = os.path.join(base_path, "data/데이터초안/전국어린이보호구역표준데이터.csv")

        # 소방서 좌표 로드
        fire_stations = []
        if os.path.exists(fire_station_file):
            try:
                df_fire = pd.read_csv(fire_station_file)
                # 좌표만 추출 (주소 정보 제외)
                for _, row in df_fire.iterrows():
                    if not pd.isna(row['위도']) and not pd.isna(row['경도']):
                        fire_stations.append([float(row['위도']), float(row['경도'])])
                print(f"✅ 소방서 좌표 {len(fire_stations)}개 로드 완료")
            except Exception as e:
                print(f"❌ 소방서 데이터 로드 오류: {e}")

        # 어린이보호구역 좌표 로드
        safety_zones = []
        if os.path.exists(child_safety_file):
            try:
                # 인코딩 문제가 있을 수 있으므로 여러 인코딩 시도
                encodings = ['utf-8', 'euc-kr', 'cp949']
                df_safety = None

                for encoding in encodings:
                    try:
                        df_safety = pd.read_csv(child_safety_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue

                if df_safety is not None:
                    # 위도, 경도 컬럼 찾기
                    lat_col = None
                    lng_col = None
                    for col in df_safety.columns:
                        if '위도' in col:
                            lat_col = col
                        elif '경도' in col:
                            lng_col = col

                    # 좌표만 추출 (주소 정보 제외)
                    for _, row in df_safety.iterrows():
                        if lat_col and lng_col and not pd.isna(row[lat_col]) and not pd.isna(row[lng_col]):
                            safety_zones.append([float(row[lat_col]), float(row[lng_col])])
                    print(f"✅ 어린이보호구역 좌표 {len(safety_zones)}개 로드 완료")
            except Exception as e:
                print(f"❌ 어린이보호구역 데이터 로드 오류: {e}")

        return jsonify({
            'fireStations': fire_stations,
            'safetyZones': safety_zones
        })

    except Exception as e:
        print(f"❌ 좌표 데이터 처리 오류: {e}")
        return jsonify({'error': str(e)}), 500


# ✅ 1. 추천 알고리즘 실행 API
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        print("📌 [1] 추천 요청 수신됨")

        # === 파일 경로 설정 ===
        base_path = os.path.dirname(__file__)
        population_density_file = os.path.join(base_path, "data/데이터초안/인구밀도 데이터.txt")
        child_safety_zone_file = os.path.join(base_path, "data/데이터초안/전국어린이보호구역표준데이터.csv")
        geo_mapping_file = os.path.join(base_path, "data/산출데이터/지역코드_좌표.csv")
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
            if any(geodesic(pop_loc, (row['위도'], row['경도'])).km < safety_distance for _, row in
                   df_child_safety.iterrows()):
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
        plt.scatter(cluster_centroids['경도'], cluster_centroids['위도'], c='red', label='Cluster Centroids', s=100,
                    marker='x')
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
@app.route('/recommend/compare', methods=['POST'])
def compare_existing_with_recommended():
    try:
        print("📌 [compare] 기존 수거함과 추천 위치 비교 시작")

        # === Spring Boot에서 기존 수거함 좌표 데이터 가져오기 ===
        spring_url = f"{BACKEND_ORIGIN}/admin/findAllBox"
        response = requests.get(spring_url, verify=False)
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


@app.route('/upload-multiple', methods=['POST'])
def upload_multiple_files():
    try:
        files = request.files
        print("📥 수신된 파일 목록:", list(files.keys()))

        save_dir = "data/데이터초안"
        os.makedirs(save_dir, exist_ok=True)

        # 🔥 기존 파일 삭제
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"🗑️ 기존 파일 삭제됨: {file_path}")
            except Exception as e:
                print(f"⚠️ 파일 삭제 오류: {file_path} / {e}")

        saved_paths = {}

        for key in files:
            file = files[key]
            save_path = os.path.join(save_dir, file.filename)

            # 인구밀도 데이터 파일 이름 변경 처리
            if key == "population":
                save_path = os.path.join(save_dir, "인구밀도 데이터.txt")  # 이름 변경
            file.save(save_path)
            saved_paths[key] = save_path
            print(f"✅ 저장됨: {key} → {save_path}")

        # 🔄 소방서 주소 → 좌표 변환 처리
        if "fireStation" in files:
            print("📌 [자동 처리] 소방서 주소 → 좌표 변환 중...")
            fire_file_path = saved_paths["fireStation"]
            df = pd.read_csv(fire_file_path, encoding="cp949")

            from time import sleep
            from tqdm import tqdm

            latitudes, longitudes, failed = [], [], []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                address = str(row["주소"]).strip()
                lat, lng = kakao_geocode(address)
                latitudes.append(lat)
                longitudes.append(lng)
                if lat is None or lng is None:
                    failed.append(address)
                sleep(0.3)

            df["위도"] = latitudes
            df["경도"] = longitudes
            os.makedirs("data/산출데이터", exist_ok=True)
            df.to_csv("data/산출데이터/소방서_좌표_카카오.csv", index=False, encoding="utf-8-sig")
            print("✅ 소방서 주소 변환 완료")

        # 🔄 SHP → 중심 좌표 추출 처리
        if "boundaryshp" in files:
            print("📌 [자동 처리] SHP → 중심 좌표 추출 중...")
            shp_path = saved_paths["boundaryshp"]
            gdf = gpd.read_file(shp_path)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs(epsg=4326)

            gdf["위도"] = gdf.geometry.centroid.y
            gdf["경도"] = gdf.geometry.centroid.x
            region_col = "TOT_REG_CD"
            df_geo = gdf[[region_col, "위도", "경도"]].rename(columns={region_col: "지역코드"})
            os.makedirs("data/산출데이터", exist_ok=True)
            df_geo.to_csv("data/산출데이터/지역코드_좌표.csv", index=False, encoding="utf-8-sig")
            print("✅ SHP 중심 좌표 추출 완료")

        # ✅ 추천 알고리즘 자동 실행
        print("🚀 [후처리] 추천 알고리즘 자동 실행 시작")
        with app.test_request_context():
            res = recommend()
            print("🎯 추천 알고리즘 자동 실행 완료")

        return jsonify({"message": "모든 파일 업로드 및 자동 변환 완료 ✅"}), 200

    except Exception as e:
        print(f"❌ 업로드 오류: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
