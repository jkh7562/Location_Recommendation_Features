import geopandas as gpd

# SHP 파일 경로
shp_file = "../data/경계데이터/bnd_oa_34040_2024_2Q.shp"

# SHP 파일 로드 (속성 데이터만 가져오기)
gdf = gpd.read_file(shp_file, ignore_geometry=True)

# SHP 파일의 컬럼명 확인
print("SHP 파일 컬럼명:", gdf.columns)

# 지역코드 데이터 샘플 확인
print(gdf.head())
