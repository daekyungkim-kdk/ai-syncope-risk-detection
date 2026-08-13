from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="웨어러블 생체신호 이상 징후 탐지",
    page_icon="⌚",
    layout="wide",
)

STAGE_KO = {
    "normal": "정상",
    "caution": "주의",
    "risk": "위험 징후",
    "check_sensor": "센서 확인",
}
STAGE_COLOR = {
    "normal": "#2e8b57",
    "caution": "#f0a202",
    "risk": "#d62828",
    "check_sensor": "#6c757d",
}


@st.cache_data
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def validate(frame: pd.DataFrame) -> list[str]:
    required = {"window_start_s", "anomaly_score", "risk_stage"}
    return sorted(required - set(frame.columns))


st.title("다중 생체신호 이상 징후 탐지 프로토타입")
st.caption("PPG · HRV · 움직임 · EDA · 피부온도 기반 개인 baseline 비교")
st.warning(
    "본 화면은 연구·교육용 프로토타입입니다. 실신을 진단하거나 예측하는 "
    "의료기기가 아니며, 위험 징후는 평상시 대비 생체신호 이상도를 의미합니다."
)

default_path = Path("outputs/s1/scored_windows.csv")
uploaded = st.sidebar.file_uploader("분석 결과 CSV 선택", type="csv")
if uploaded is not None:
    frame = read_csv(uploaded)
    source_name = uploaded.name
elif default_path.exists():
    frame = read_csv(default_path)
    source_name = str(default_path)
else:
    st.info(
        "왼쪽에서 scored_windows.csv 파일을 선택하거나, 먼저 다음 명령으로 분석을 실행하세요."
    )
    st.code(
        r".\.venv\Scripts\python.exe scripts\train_prototype.py "
        r"--features data\ppg_dalia_s1_features.csv --output-dir outputs\s1",
        language="powershell",
    )
    st.stop()

missing = validate(frame)
if missing:
    st.error("필수 열이 없습니다: " + ", ".join(missing))
    st.stop()

frame = frame.copy()
frame["risk_stage"] = frame["risk_stage"].astype(str)
frame["단계"] = frame["risk_stage"].map(STAGE_KO).fillna(frame["risk_stage"])
frame["시간(분)"] = pd.to_numeric(frame["window_start_s"], errors="coerce") / 60
frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="coerce")

st.sidebar.success(f"불러온 파일: {source_name}")
if "subject_id" in frame:
    subjects = frame["subject_id"].dropna().astype(str).unique().tolist()
    chosen = st.sidebar.selectbox("피험자", subjects)
    view = frame[frame["subject_id"].astype(str) == chosen].copy()
else:
    view = frame

counts = view["risk_stage"].value_counts()
columns = st.columns(5)
columns[0].metric("분석 구간", f"{len(view):,}")
for column, stage in zip(columns[1:], ("normal", "caution", "risk", "check_sensor")):
    column.metric(STAGE_KO[stage], f"{int(counts.get(stage, 0)):,}")

st.subheader("시간별 이상도")
score_chart = px.line(
    view,
    x="시간(분)",
    y="anomaly_score",
    color="단계",
    color_discrete_map={STAGE_KO[k]: v for k, v in STAGE_COLOR.items()},
    markers=True,
    labels={"anomaly_score": "이상도(0~1)"},
)
score_chart.add_hline(y=0.45, line_dash="dot", line_color="#f0a202", annotation_text="주의 기준")
score_chart.add_hline(y=0.70, line_dash="dash", line_color="#d62828", annotation_text="위험 기준")
st.plotly_chart(score_chart, use_container_width=True)

left, right = st.columns(2)
if "hr_bpm" in view:
    with left:
        st.subheader("심박수")
        st.plotly_chart(
            px.line(view, x="시간(분)", y="hr_bpm", labels={"hr_bpm": "HR (bpm)"}),
            use_container_width=True,
        )

hrv_columns = [name for name in ("sdnn_ms", "rmssd_ms") if name in view]
if hrv_columns:
    with right:
        st.subheader("HRV")
        hrv = view.melt(
            id_vars="시간(분)", value_vars=hrv_columns,
            var_name="HRV 지표", value_name="값(ms)",
        )
        st.plotly_chart(
            px.line(hrv, x="시간(분)", y="값(ms)", color="HRV 지표"),
            use_container_width=True,
        )

st.subheader("주의가 필요한 구간")
alerts = view[view["risk_stage"].isin(["caution", "risk", "check_sensor"])].copy()
display_columns = [
    name for name in (
        "시간(분)", "window_end_s", "단계", "anomaly_score", "hr_bpm",
        "sdnn_ms", "rmssd_ms", "ppg_sqi",
    ) if name in alerts
]
if alerts.empty:
    st.success("선택한 데이터에서 주의 구간이 없습니다.")
else:
    st.dataframe(
        alerts[display_columns].sort_values("anomaly_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    if (alerts["risk_stage"] == "risk").any():
        st.error("위험 징후 구간: 실제 사용 환경에서는 앉거나 안전한 장소로 이동하도록 안내합니다.")
    if (alerts["risk_stage"] == "check_sensor").any():
        st.info("센서 확인 구간: 워치 착용 상태와 PPG 신호 품질을 먼저 확인해야 합니다.")

st.download_button(
    "현재 결과 CSV 다운로드",
    data=view.to_csv(index=False).encode("utf-8-sig"),
    file_name="scored_windows_dashboard.csv",
    mime="text/csv",
)
