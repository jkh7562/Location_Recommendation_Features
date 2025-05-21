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
CORS(app, resources={r"/*": {"origins": BACKEND_ORIGIN}})


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


# ✅ 1. 추천 알고리즘 실행API
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
        # 모든 지역 포함
        df_population_filtered = df_population.copy()

        print("📌 [3] 위경도 매핑 중...")
        df_geo = pd.read_csv(geo_mapping_file).dropna(subset=['위도', '경도'])
        df_population_filtered = df_population_filtered.merge(df_geo, on="지역코드", how="left")

        print("📌 [4] 고밀도 지역 필터링 중...")
        df_population_filtered['인구밀도평균'] = df_population_filtered['인구밀도'].rolling(window=5, center=True, min_periods=1).mean()
        df_population_filtered['밀도차이'] = df_population_filtered['인구밀도'] - df_population_filtered['인구밀도평균']
        density_threshold = 0.8
        high_density_areas = df_population_filtered[
            df_population_filtered['밀도차이'] > df_population_filtered['밀도차이'].quantile(density_threshold)].copy()

        min_recommendations = 20
        while len(high_density_areas) < min_recommendations and density_threshold > 0.5:
            density_threshold -= 0.05
            high_density_areas = df_population_filtered[
                df_population_filtered['밀도차이'] > df_population_filtered['밀도차이'].quantile(density_threshold)].copy()

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
        dbscan = DBSCAN(eps=epsilon_radians, min_samples=2, metric='haversine')
        df_final_filtered['cluster'] = dbscan.fit_predict(coords)

        # 군집화되지 않은 좌표 (노이즈)
        df_noise = df_final_filtered[df_final_filtered['cluster'] == -1].copy()
        df_noise['point_type'] = 'noise'  # 군집화되지 않은 포인트 표시

        # 군집화된 좌표들
        df_clustered = df_final_filtered[df_final_filtered['cluster'] != -1].copy()
        df_clustered['point_type'] = 'cluster_member'  # 군집에 포함된 포인트 표시

        # 군집 중심점 계산
        cluster_centroids = df_clustered.groupby('cluster').agg({'위도': 'mean', '경도': 'mean'}).reset_index()
        cluster_centroids['point_type'] = 'centroid'  # 군집 중심점 표시

        # 모든 데이터를 하나의 DataFrame으로 합치기
        final_recommendations = pd.concat([
            cluster_centroids[['위도', '경도', 'cluster', 'point_type']],
            df_clustered[['위도', '경도', 'cluster', 'point_type']],
            df_noise[['위도', '경도', 'cluster', 'point_type']]
        ])

        print(f"✅ 군집화 완료 - 군집 수: {df_clustered['cluster'].nunique()}, 노이즈 수: {len(df_noise)}")

        print("📌 [8] 결과 CSV 저장 중...")
        final_recommendations.to_csv(result_output_file, index=False, encoding='utf-8-sig')
        print(f"🎯 추천 완료! 위치 수: {len(final_recommendations)}")
        print(f"💾 저장 경로: {result_output_file}")

        print("📌 [9] 시각화 이미지 저장 중...")
        plt.figure(figsize=(10, 6))
        # 군집화되지 않은 좌표 (노이즈)
        noise_points = final_recommendations[final_recommendations['point_type'] == 'noise']
        plt.scatter(noise_points['경도'], noise_points['위도'], c='green', label='Non-clustered (Noise)', s=50)

        # 군집에 포함된 좌표들
        cluster_members = final_recommendations[final_recommendations['point_type'] == 'cluster_member']
        plt.scatter(cluster_members['경도'], cluster_members['위도'], c='blue', label='Cluster Members', s=50)

        # 군집 중심점
        centroids = final_recommendations[final_recommendations['point_type'] == 'centroid']
        plt.scatter(centroids['경도'], centroids['위도'], c='red', label='Cluster Centroids', s=100, marker='x')

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


@app.route('/recommend/compare', methods=['POST'])
def compare_existing_with_recommended():
    try:
        print("📌 [compare] 기존 수거함과 추천 위치 비교 시작")

        # === 요청 본문에서 기존 수거함 데이터 가져오기 ===
        if not request.is_json:
            print("❌ 요청 본문이 JSON 형식이 아닙니다.")
            return jsonify({"error": "요청 본문이 JSON 형식이어야 합니다."}), 400

        existing_boxes_raw = request.json
        print(f"📦 기존 수거함 응답 수: {len(existing_boxes_raw)}")

        # 중요: 응답 구조 확인 및 처리
        # 응답이 {'box': {...}} 형태로 중첩되어 있으므로 'box' 키 내부의 데이터를 추출
        existing_boxes = []
        for item in existing_boxes_raw:
            if isinstance(item, dict) and 'box' in item:
                existing_boxes.append(item['box'])
            else:
                existing_boxes.append(item)  # 중첩되지 않은 경우 그대로 사용

        print(f"📦 처리된 기존 수거함 수: {len(existing_boxes)}")

        # 첫 5개 박스 데이터 로깅 (또는 전체가 5개 미만이면 전체)
        sample_size = min(5, len(existing_boxes))
        print(f"📋 기존 수거함 샘플 데이터 ({sample_size}개):")
        for i in range(sample_size):
            print(f"  - Box {i + 1}: {existing_boxes[i]}")

        # === 기존 수거함의 좌표 파싱 ===
        existing_coords = []
        valid_coords_count = 0
        invalid_coords_count = 0

        print("🧮 기존 수거함 좌표 파싱 시작...")
        for idx, box in enumerate(existing_boxes):
            location_str = box.get("location")
            box_id = box.get("id", "알 수 없음")
            box_name = box.get("name", "이름 없음")

            if not location_str:
                print(f"⚠️ 위치 정보 없음: Box ID {box_id}, Name: {box_name}")
                invalid_coords_count += 1
                continue

            if "POINT" not in location_str:
                print(f"⚠️ POINT 형식이 아님: Box ID {box_id}, Name: {box_name}, Location: {location_str}")
                invalid_coords_count += 1
                continue

            try:
                # 좌표 파싱 시도
                coords_part = location_str.replace("POINT (", "").replace(")", "").strip()
                lng, lat = map(float, coords_part.split())
                existing_coords.append((lat, lng))  # 위도, 경도
                valid_coords_count += 1

                # 처음 10개와 마지막 10개 좌표만 출력 (또는 전체가 20개 미만이면 전체)
                if idx < 10 or idx >= len(existing_boxes) - 10:
                    print(f"  ✓ Box {idx + 1} (ID: {box_id}): 위도={lat}, 경도={lng}, 원본={location_str}")
            except ValueError as e:
                print(f"⚠️ 위치 파싱 실패: Box ID {box_id}, Name: {box_name}, Location: {location_str}, 오류: {e}")
                invalid_coords_count += 1

        print(f"📊 좌표 파싱 결과: 성공={valid_coords_count}, 실패={invalid_coords_count}, 총={len(existing_boxes)}")
        print(f"🗺️ 유효한 기존 수거함 좌표 수: {len(existing_coords)}")

        # === 추천 위치 CSV 불러오기 ===
        base_path = os.path.dirname(__file__)
        recommended_file = os.path.join(base_path, "data/산출데이터/추천_수거함_위치.csv")
        print(f"📂 추천 위치 CSV 파일 경로: {recommended_file}")

        if not os.path.exists(recommended_file):
            print(f"❌ 추천 위치 CSV 파일이 존재하지 않습니다: {recommended_file}")
            return jsonify({"error": "추천 위치 CSV 파일을 찾을 수 없습니다."}), 404

        df_recommended = pd.read_csv(recommended_file)
        print(f"📍 추천 위치 수 (CSV): {len(df_recommended)}")
        print(f"📋 추천 위치 CSV 컬럼: {df_recommended.columns.tolist()}")

        # 군집 중심점과 노이즈 포인트 필터링 (실제 추천 위치로 사용)
        df_centroids_noise = df_recommended[
            (df_recommended['point_type'] == 'centroid') |
            (df_recommended['point_type'] == 'noise')
            ]
        print(f"📍 중심점 및 노이즈 포인트 수: {len(df_centroids_noise)}")
        print(f"  - 중심점(centroid) 수: {len(df_recommended[df_recommended['point_type'] == 'centroid'])}")
        print(f"  - 노이즈(noise) 수: {len(df_recommended[df_recommended['point_type'] == 'noise'])}")

        # === 비교: 기존 수거함과 400m 이내인 추천 위치 제거 ===
        filtered_centroids_noise = []
        valid_clusters = set()  # 유효한 군집 ID를 저장할 집합
        removed_count = 0  # 삭제된 위치 개수 카운트
        removed_details = []  # 삭제된 위치의 상세 정보

        print("🔍 기존 수거함과 추천 위치 비교 시작...")
        for idx, row in df_centroids_noise.iterrows():
            rec_point = (row["위도"], row["경도"])
            too_close = False
            closest_distance = float('inf')
            closest_existing_point = None

            for exist_point in existing_coords:
                distance = geodesic(rec_point, exist_point).km
                if distance < closest_distance:
                    closest_distance = distance
                    closest_existing_point = exist_point

                if distance < 0.4:  # 400m 이내
                    too_close = True
                    removed_count += 1  # 삭제된 위치 카운트 증가
                    removed_details.append({
                        "추천_위도": rec_point[0],
                        "추천_경도": rec_point[1],
                        "기존_위도": exist_point[0],
                        "기존_경도": exist_point[1],
                        "거리_km": distance,
                        "point_type": row["point_type"],
                        "cluster": int(row["cluster"])
                    })
                    break

            if not too_close:
                cluster_id = int(row["cluster"])
                filtered_centroids_noise.append({
                    "lat": rec_point[0],
                    "lng": rec_point[1],
                    "point_type": row["point_type"],
                    "cluster": cluster_id
                })

                # 유효한 군집 ID 저장 (노이즈 포인트는 -1이므로 제외)
                if cluster_id != -1:
                    valid_clusters.add(cluster_id)

            # 처리 진행 상황 로깅 (10% 단위)
            if idx % max(1, len(df_centroids_noise) // 10) == 0:
                print(f"  - 처리 중: {idx}/{len(df_centroids_noise)} ({idx / len(df_centroids_noise) * 100:.1f}%)")

        # 삭제된 위치 상세 정보 로깅 (최대 10개)
        print(f"🗑️ 삭제된 추천 위치 상세 정보 (최대 10개):")
        for i, detail in enumerate(removed_details[:10]):
            print(f"  - 삭제 {i + 1}: 추천({detail['추천_위도']}, {detail['추천_경도']}) ↔ " +
                  f"기존({detail['기존_위도']}, {detail['기존_경도']}), " +
                  f"거리={detail['거리_km']:.3f}km, 타입={detail['point_type']}, 군집={detail['cluster']}")

        # 군집 멤버 추가 (유효한 군집에 속한 멤버만)
        cluster_members = []
        df_members = df_recommended[df_recommended['point_type'] == 'cluster_member']
        print(f"👥 군집 멤버 수 (전체): {len(df_members)}")

        for _, row in df_members.iterrows():
            cluster_id = int(row["cluster"])
            if cluster_id in valid_clusters:  # 유효한 군집에 속한 멤버만 추가
                cluster_members.append({
                    "lat": row["위도"],
                    "lng": row["경도"],
                    "point_type": "cluster_member",
                    "cluster": cluster_id
                })

        # 모든 결과 합치기
        all_recommendations = filtered_centroids_noise + cluster_members

        print(f"✅ 필터링 후 추천 위치 수: {len(filtered_centroids_noise)} (중심점 및 노이즈)")
        print(f"  - 중심점(centroid): {len([p for p in filtered_centroids_noise if p['point_type'] == 'centroid'])}")
        print(f"  - 노이즈(noise): {len([p for p in filtered_centroids_noise if p['point_type'] == 'noise'])}")
        print(f"✅ 삭제된 추천 위치 수: {removed_count}")  # 삭제된 위치 수 출력
        print(f"✅ 포함된 군집 멤버 수: {len(cluster_members)}")
        print(f"✅ 총 반환 좌표 수: {len(all_recommendations)}")

        # 삭제된 위치 수를 응답 헤더에 포함
        response = jsonify(all_recommendations)
        response.headers['X-Removed-Locations'] = str(removed_count)
        return response, 200

    except Exception as e:
        import traceback
        print(f"❌ 오류 발생: {e}")
        print(f"❌ 상세 오류: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload-multiple', methods=['POST'])
def upload_multiple_files():
    try:
        files = request.files
        print(f"📥 수신된 파일 목록: {list(files.keys())}")

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

        for field_name in files:
            file = files[field_name]
            save_path = os.path.join(save_dir, file.filename)

            # 인구밀도 데이터 파일 이름 변경 처리
            if field_name == "population":
                save_path = os.path.join(save_dir, "인구밀도 데이터.txt")
            file.save(save_path)
            saved_paths[field_name] = save_path
            print(f"✅ 저장됨: {field_name} → {save_path}, 실제 파일명: {file.filename}")  # 실제 파일명 로그 추가

        # 파일 경로 확인 로그 추가
        population_file_path = "data/데이터초안/인구밀도 데이터.txt"
        print(f"🔍 인구밀도 파일 경로 확인: {population_file_path}")
        print(f"   존재 여부: {os.path.exists(population_file_path)}")
        print(f"   파일 여부: {os.path.isfile(population_file_path)}")

        # 🔄 소방서 주소 → 좌표 변환 처리
        if "fireStation" in saved_paths:
            print("📌 [자동 처리] 소방서 주소 → 좌표 변환 중...")
            fire_file_path = saved_paths["fireStation"]
            df = pd.read_csv(fire_file_path, encoding="cp949")

            from time import sleep

            latitudes, longitudes, failed = [], [], []
            total = len(df)
            for idx, row in enumerate(df.iterrows()):
                _, row_data = row
                address = str(row_data["주소"]).strip()
                lat, lng = kakao_geocode(address)
                latitudes.append(lat)
                longitudes.append(lng)
                if lat is None or lng is None:
                    failed.append(address)

                if (idx + 1) % 10 == 0 or idx == total - 1:
                    print(f"⏳ 주소 변환 진행 중: {idx + 1}/{total} ({((idx + 1) / total * 100):.1f}%)")

                sleep(0.3)

            df["위도"] = latitudes
            df["경도"] = longitudes
            os.makedirs("data/산출데이터", exist_ok=True)
            df.to_csv("data/산출데이터/소방서_좌표_카카오.csv", index=False, encoding="utf-8-sig")
            print("✅ 소방서 주소 변환 완료")

        # 🔄 SHP → 중심 좌표 추출 처리
        if "boundaryshp" in saved_paths:
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

@app.route('/server-status', methods=['GET'])
def check_server_status():
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
