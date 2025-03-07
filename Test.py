import geopandas as gpd

# SHP 파일 경로
shp_file = "data/bnd_oa_34040_2024_2Q.shp"

# SHP 파일 로드
gdf = gpd.read_file(shp_file)

# 데이터프레임의 컬럼명 출력
print("SHP 파일의 컬럼명 확인:", gdf.columns)
print("SHP 파일 일부 데이터 확인:", gdf.head())
