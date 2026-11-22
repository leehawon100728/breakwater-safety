"""피규어 YOLO 모델 연동 — 위험 피규어(danger_figure) + 인원(figure_people).

카메라 최신 프레임을 두 모델에 넣고, 관제 화면(index.html)이 이미 읽는 /api/status 형식으로 바꿔 보관한다.
  analysis.danger : 위험 피규어가 안정적으로 보이면 positive=True
  analysis.people : 인원(최근 프레임의 다수결)
  analysis.fall   : 추락은 danger_figure 가 '위험'에 포함해 구분하지 않으므로 판정하지 않음 (no_target)
  persons         : 피규어 박스(0~1 비율 [x, y, w, h]) + 번호
설정: config.json 의 "ai_engine": "figure" (기본) / "teachable" (예전 Teachable Machine 3모델)
"""
import logging
import threading
import time
from collections import Counter, deque

import numpy as np

log = logging.getLogger("ai")


class FigureEngine:
    """두 모델을 한 번만 불러와 모든 카메라가 공유한다 (YOLO 추론은 동시에 부르지 않도록 잠금)."""

    def __init__(self, conf_danger=0.5, conf_people=0.25):
        self.conf_danger = conf_danger
        self.conf_people = conf_people
        self.lock = threading.Lock()
        self.errors = {"danger": "", "people": ""}
        self._danger = self._people = None
        try:
            from danger_figure import predict_danger
            self._danger = predict_danger
        except Exception as e:
            self.errors["danger"] = f"{type(e).__name__}: {e}"
        try:
            from figure_people import predict_safe_people
            self._people = predict_safe_people
        except Exception as e:
            self.errors["people"] = f"{type(e).__name__}: {e}"

    def info(self, key):
        err = self.errors[key]
        return {"folder": "danger_figure" if key == "danger" else "figure_people", "loaded": not err,
                "labels": ["위험", "안전"] if key == "danger" else ["person_figure"], "error": err, "input": None}

    def run(self, frame):
        """(위험 결과, 인원 결과) — 모델 하나가 불러지지 않았으면 그 자리는 None."""
        with self.lock:
            d = self._danger(frame, conf=self.conf_danger) if self._danger else None
            p = self._people(frame, confidence=self.conf_people) if self._people else None
        return d, p


class _Det:  # app.py 의 status 가 읽는 모양 (t.det.error / t.det.kind)
    kind = "figure"

    def __init__(self, engine):
        self.engine = engine

    @property
    def error(self):
        return self.engine.errors["people"]


class _PersonsView:
    def __init__(self, analyzer):
        self.a = analyzer
        self.det = analyzer.det

    @property
    def result(self):
        return self.a.persons


class FigureAnalyzer(threading.Thread):
    """카메라 하나의 분석 스레드. result(분석)와 persons(박스)를 갱신한다."""

    def __init__(self, capture, engine, acfg):
        super().__init__(daemon=True)
        self.cap = capture
        self.engine = engine
        self.det = _Det(engine)
        self.interval = float(acfg.get("interval_sec", 0.3))
        win = int(acfg.get("window", 8))
        self.agree = float(acfg.get("min_agreement", 0.6))
        self.d_hist = deque(maxlen=win)  # (위험 박스 수, 위험 최고 신뢰도)
        self.p_hist = deque(maxlen=win)  # (인원 수, 평균 신뢰도)
        self.tracks = {}                 # 번호 → 중심점 (같은 피규어에 같은 번호)
        self.last_seq = -1
        self.last_log = None
        self.result = {}
        self.persons = {"state": "pending", "error": "", "count": None, "boxes": []}

    def tracker(self):
        """app.py 의 TRACKERS 에 넣는 어댑터 (persons_of 가 t.result / t.det.error 를 읽음)."""
        return _PersonsView(self)

    def _clear(self):
        self.d_hist.clear()
        self.p_hist.clear()
        self.tracks.clear()

    def _number(self, cx, cy, used):
        """가장 가까운 이전 중심점의 번호를 잇고, 없으면 가장 작은 빈 번호를 준다."""
        best, bd = None, 0.12  # 화면 폭의 12% 이내만 같은 피규어로 봄
        for no, (px, py) in self.tracks.items():
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if no not in used and d < bd:
                best, bd = no, d
        if best is None:
            best = 1
            while best in used or best in self.tracks:
                best += 1
        return best

    def run(self):
        while True:
            t0 = time.time()
            frame, seq, ts = self.cap.latest()
            if frame is None or time.time() - ts > 3:
                self._clear()  # 영상이 끊기면 이전 판단을 지금 결과처럼 쓰지 않음
                self.persons = {"state": "pending", "error": "", "count": None, "boxes": [], "ts": time.time()}
                time.sleep(0.2)
                continue
            if seq == self.last_seq:
                time.sleep(0.02)
                continue
            self.last_seq = seq
            if float(frame.mean()) < 12 and int(frame.max()) < 40:  # 카메라 가림·신호 없음
                self._clear()
                self.result = {"ts": time.time(), "frame_ts": ts, "error": "", "dark": True,
                               **{k: {"state": "no_signal", "label": None, "confidence": None} for k in ("people", "danger", "fall")}}
                time.sleep(self.interval)
                continue
            try:
                d, p = self.engine.run(frame)
                self._publish(frame, ts, d, p)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                log.exception("%s 피규어 분석 오류", self.cap.id)
                self.result = {"ts": time.time(), "frame_ts": ts, "error": msg,
                               **{k: {"state": "error", "error": msg} for k in ("people", "danger", "fall")}}
                self.persons = {"state": "error", "error": msg, "count": None, "boxes": [], "ts": time.time()}
                time.sleep(1)
            time.sleep(max(0.0, self.interval - (time.time() - t0)))

    def _publish(self, frame, ts, d, p):
        h, w = frame.shape[:2]
        now = time.time()
        out = {"ts": now, "frame_ts": ts, "error": ""}

        # 위험: 최근 프레임 중 위험 피규어가 보인 비율로 안정화
        if d is None:
            out["danger"] = {"state": "error", "error": self.engine.errors["danger"] or "모델 없음"}
        else:
            top = max((x["confidence"] for x in d["persons"]), default=0.0)
            self.d_hist.append((d["danger_count"], top))
            out["danger"] = self._danger_state(d)

        # 인원: 최근 프레임의 최빈값
        boxes = []
        if p is None:
            out["people"] = {"state": "error", "error": self.engine.errors["people"] or "모델 없음"}
            self.persons = {"state": "error", "error": self.engine.errors["people"], "count": None, "boxes": [], "ts": now}
        else:
            confs = [x["confidence"] for x in p["persons"]]
            self.p_hist.append((p["total_count"], float(np.mean(confs)) if confs else 1.0))
            out["people"] = self._people_state(p)
            used, tracks = set(), {}
            for x in sorted(p["persons"], key=lambda q: q["bbox"][0]):
                x1, y1, x2, y2 = x["bbox"]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                no = self._number(cx, cy, used)
                used.add(no)
                tracks[no] = (cx, cy)
                boxes.append({"id": no, "label": f"PERSON {no}", "box": [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)],
                              "conf": x["confidence"], "held": False})
            self.tracks = tracks
            boxes.sort(key=lambda b: b["id"])
            self.persons = {"state": "ok", "mode": "figure", "error": "", "count": len(boxes), "boxes": boxes,
                            "frame_size": [w, h], "frame_ts": ts, "ts": now}

        out["fall"] = {"state": "no_target", "label": None, "confidence": None, "positive": None,
                       "note": "추락은 위험 판정에 포함 (구분하지 않음)"}
        self.result = out

    def _danger_state(self, d):
        n = len(self.d_hist)
        if n < min(3, self.d_hist.maxlen):
            return {"state": "pending", "label": None, "confidence": None}
        hits = [c for k, c in self.d_hist if k > 0]
        frac = len(hits) / n
        raw = {"label": "위험" if d["danger_count"] else "안전", "count": d["danger_count"]}
        if frac >= self.agree:
            return {"state": "ok", "label": "위험", "confidence": round(float(np.mean(hits)), 3), "positive": True, "count": d["danger_count"], "raw": raw}
        if 1 - frac >= self.agree:
            return {"state": "ok", "label": "안전", "confidence": round(1 - frac, 3), "positive": False, "count": 0, "raw": raw}
        return {"state": "uncertain", "label": None, "candidate": "위험" if frac >= 0.5 else "안전",
                "confidence": round(max(frac, 1 - frac), 3), "positive": None, "raw": raw}

    def _people_state(self, p):
        n = len(self.p_hist)
        if n < min(3, self.p_hist.maxlen):
            return {"state": "pending", "label": None, "confidence": None, "count": None}
        count, votes = Counter(c for c, _ in self.p_hist).most_common(1)[0]
        if votes / n < self.agree:
            return {"state": "uncertain", "label": None, "candidate": f"{count}명", "confidence": round(votes / n, 3), "count": None}
        conf = float(np.mean([c for k, c in self.p_hist if k == count]))
        return {"state": "ok", "label": f"{count}명", "confidence": round(conf, 3), "count": count,
                "raw": {"label": f"{p['total_count']}명", "confidence": round(conf, 3)}}
