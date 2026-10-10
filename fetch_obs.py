"""기상청 API허브 AWS 매분자료에서 부산 관측소 값을 받아 obs.json으로 저장한다."""
import json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

STATIONS = {"937": "해운대", "942": "부산남구", "904": "사하", "921": "가덕도", "910": "영도", "159": "부산"}
KST = timezone(timedelta(hours=9))
key = os.environ["KMA_KEY"]

def val(text):
    v = float(text)
    return None if v <= -50 else v  # -50 이하는 결측

for back in (5, 10, 20):  # 최신 분 자료가 아직 없거나 응답이 늦으면 조금 앞 시각으로 재시도
    tm = (datetime.now(KST) - timedelta(minutes=back)).strftime("%Y%m%d%H%M")
    url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm2={tm}&stn=0&disp=1&help=0&authKey={key}"
    try:
        raw = urllib.request.urlopen(url, timeout=60).read().decode("euc-kr", "ignore")
    except OSError as e:
        print(tm, "요청 실패:", type(e).__name__)
        continue
    out = {}
    for line in raw.splitlines():
        f = line.split(",")
        if len(f) > 12 and f[1] in STATIONS:
            out[f[1]] = {"name": STATIONS[f[1]], "wind": val(f[7]), "gust": val(f[5]), "rain": val(f[11]), "temp": val(f[8]), "hum": val(f[14])}
    if out:
        json.dump({"time": tm, "stations": out}, open("obs.json", "w"), ensure_ascii=False)
        print(tm, out)
        sys.exit(0)
sys.exit("관측값을 받지 못했습니다")
