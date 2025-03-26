import requests

url = "http://localhost:8081/admin/findAllBox"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("✅ 데이터 수신 성공:", data)
else:
    print("❌ 실패:", response.status_code)
