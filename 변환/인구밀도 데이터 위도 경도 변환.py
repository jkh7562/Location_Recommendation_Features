import geopandas as gpd
import pandas as pd

# SHP 파일 경로
shp_file = "../data/데이터초안/bnd_oa_34040_2024_2Q.shp"

# SHP 파일 로드
gdf = gpd.read_file(shp_file)

# 좌표계를 WGS84 (EPSG:4326)로 변환 (SGIS 기본 좌표계는 다를 수 있음)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# SHP 파일의 컬럼명 출력하여 확인
print("SHP 파일 컬럼명:", gdf.columns)

# 집계구 지역코드 컬럼 지정 (TOT_REG_CD 사용)
region_col = "TOT_REG_CD"  # 14자리 집계구별 지역코드 사용

# 폴리곤 중심 좌표(위도, 경도) 계산
gdf["위도"] = gdf.geometry.centroid.y
gdf["경도"] = gdf.geometry.centroid.x

# 필요한 컬럼만 선택 (지역코드를 변환하지 않고 그대로 사용)
df_geo = gdf[[region_col, "위도", "경도"]].rename(columns={region_col: "지역코드"})

# CSV로 저장 (집계구별 14자리 지역코드를 유지)
df_geo.to_csv("아산시_지역코드_좌표.csv", index=False, encoding="utf-8-sig")

# 결과 확인
print(df_geo.head())
