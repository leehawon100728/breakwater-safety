"""기상청 API허브 AWS 매분자료에서 부산 관측소 값을 받아 obs.json으로 저장한다."""
import json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

STATIONS = {"937": "해운대", "942": "부산남구", "904": "사하", "921": "가덕도", "910": "영도", "159": "부산"}
KST = timezone(timedelta(hours=9))
key = os.environ["KMA_KEY"]

def val(text):
    v = float(text)
    return None if v <= -50 else v  # -50 이하는 결측

# 10분 단위 시각만 요청한다 (그 외 시각을 전체 관측소로 요청하면 응답이 오지 않음)
now = datetime.now(KST) - timedelta(minutes=3)
base = now.replace(minute=now.minute // 10 * 10, second=0, microsecond=0)
for step in range(3):  # 최신 자료가 아직 없으면 10분씩 앞 시각으로 재시도
    tm = (base - timedelta(minutes=10 * step)).strftime("%Y%m%d%H%M")
    url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm2={tm}&stn=0&disp=1&help=0&authKey={key}"
    try:
        raw = urllib.request.urlopen(url, timeout=25).read().decode("euc-kr", "ignore")
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
