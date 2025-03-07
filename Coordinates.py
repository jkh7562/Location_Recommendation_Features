import geopandas as gpd
import pandas as pd

# SHP 파일 경로
shp_file = "data/bnd_oa_34040_2024_2Q.shp"

# SHP 파일 로드
gdf = gpd.read_file(shp_file)

# 좌표계를 WGS84 (EPSG:4326)로 변환 (SGIS 기본 좌표계는 다를 수 있음)
gdf = gdf.to_crs(epsg=4326)

# SHP 파일의 컬럼명 확인 후, 지역코드 컬럼 선택
print("SHP 파일의 컬럼명:", gdf.columns)

# 사용 가능한 지역코드 컬럼 자동 탐색
possible_region_cols = ["OA_CD", "adm_cd", "gid", "SGG_CD", "EMD_CD", "BJD_CD"]
region_col = None

for col in possible_region_cols:
    if col in gdf.columns:
        region_col = col
        break

if region_col is None:
    raise KeyError("SHP 파일에서 지역코드 컬럼을 찾을 수 없습니다.")

# 지역코드 + 중심 좌표(위도, 경도) 추출
gdf["위도"] = gdf.geometry.centroid.y
gdf["경도"] = gdf.geometry.centroid.x

# 필요한 컬럼만 선택
df_geo = gdf[[region_col, "위도", "경도"]].rename(columns={region_col: "지역코드"})

# 결과 확인
print(df_geo.head())

# CSV로 저장 (활용 가능)
df_geo.to_csv("아산시_지역코드_좌표.csv", index=False, encoding="utf-8-sig")
print("위도/경도 데이터가 CSV 파일로 저장되었습니다.")
