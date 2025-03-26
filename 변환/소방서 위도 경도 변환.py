import requests
import pandas as pd
import time
from tqdm import tqdm  # 설치: pip install tqdm

# 📌 Kakao REST API Key
KAKAO_API_KEY = "a3eeb1b4ef391f6495af9674ae083e2d"

# 📌 주소 → 좌표 변환 함수
def kakao_geocode(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    params = {
        "query": address
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        result = response.json()
        if result["documents"]:
            first = result["documents"][0]
            return float(first["y"]), float(first["x"])  # 위도, 경도
        else:
            return None, None
    except Exception as e:
        print(f"❌ 에러: {address} / {e}")
        return None, None

# 📌 CSV 파일 로드
df = pd.read_csv("../data/데이터초안/소방청_119안전센터 현황_20240630.csv", encoding="cp949")

# 📌 위경도 컬럼 생성
latitudes = []
longitudes = []
failed = []

print("📍 주소 변환 진행 중...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    address = str(row["주소"]).strip()
    lat, lng = kakao_geocode(address)
    latitudes.append(lat)
    longitudes.append(lng)
    if lat is None or lng is None:
        failed.append(address)
    time.sleep(0.3)

df["위도"] = latitudes
df["경도"] = longitudes

# 📌 결과 저장
df.to_csv("소방서_좌표_카카오.csv", index=False, encoding="utf-8-sig")
print("✅ 변환 완료: 소방서_좌표_카카오.csv")

# 📌 실패 목록 출력
if failed:
    print("\n🚫 변환 실패 주소:")
    for addr in failed:
        print(" -", addr)
else:
    print("🎉 모든 주소 변환 성공")
