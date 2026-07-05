import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
from main import process_video

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    layout="wide"
)

# -------------------------------
# ADVANCED UI STYLE
# -------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

/* HEADER */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #00e5ff;
    margin-bottom: 25px;
}

/* METRIC CARDS */
.card {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    color: white;
    backdrop-filter: blur(12px);
    box-shadow: 0px 0px 20px rgba(0,255,255,0.2);
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.03);
}

.green {
    border-left: 6px solid #00ff88;
}

.red {
    border-left: 6px solid #ff4d4d;
}

.blue {
    border-left: 6px solid #00d4ff;
}

.orange {
    border-left: 6px solid orange;
}

.status-normal {
    color: #00ff88;
    font-size: 18px;
    font-weight: bold;
}

.status-alert {
    color: red;
    font-size: 18px;
    font-weight: bold;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    '<div class="title">🚀 AI Smart Surveillance Dashboard</div>',
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("🎛 Control Panel")

mode = st.sidebar.radio(
    "Select Input Source",
    ["📁 Upload Video", "📷 Webcam"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

# -------------------------------
# VIDEO SOURCE
# -------------------------------
video_path = None

if mode == "📷 Webcam":

    video_path = "webcam"

elif uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(uploaded_file.getvalue())

    video_path = temp_file.name

# -------------------------------
# START BUTTON
# -------------------------------
start = st.sidebar.button("▶ Start Detection")

# -------------------------------
# DASHBOARD LAYOUT
# -------------------------------
col1, col2 = st.columns([3, 1])

video_placeholder = col1.empty()

chart_placeholder = col2.empty()

m1, m2, m3, m4, m5 = st.columns(5)

# -------------------------------
# CHART
# -------------------------------
chart_data = pd.DataFrame({
    "IN": [0],
    "OUT": [0],
    "LIVE": [0]
})

chart = chart_placeholder.line_chart(chart_data)

# -------------------------------
# REPORT STORAGE
# -------------------------------
report_data = []

# -------------------------------
# MAIN PROCESS
# -------------------------------
if start:

    if video_path is None:
        st.warning("⚠ Please upload a video or select webcam.")
        st.stop()

    st.success("🟢 Detection Started")

    try:

        for frame, in_count, out_count, live, total_detected in process_video(video_path):

            # -------------------------------
            # VIDEO DISPLAY
            # -------------------------------
            video_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

            # -------------------------------
            # CROWD ALERT
            # -------------------------------
            if live > 5:
                status = "🚨 CROWD ALERT"
                status_class = "status-alert"
            else:
                status = "✅ NORMAL"
                status_class = "status-normal"

            # -------------------------------
            # IN CARD
            # -------------------------------
            m1.markdown(
                f"""
                <div class="card green">
                    <h3>🟢 IN</h3>
                    <h1>{in_count}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------
            # OUT CARD
            # -------------------------------
            m2.markdown(
                f"""
                <div class="card red">
                    <h3>🔴 OUT</h3>
                    <h1>{out_count}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------
            # LIVE CARD
            # -------------------------------
            m3.markdown(
                f"""
                <div class="card blue">
                    <h3>🔵 LIVE</h3>
                    <h1>{live}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------
            # STATUS CARD
            # -------------------------------
            m4.markdown(
                f"""
                <div class="card orange">
                    <h3>STATUS</h3>
                    <p class="{status_class}">
                        {status}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------
            # TOTAL CARD
            # -------------------------------
            m5.markdown(
                f"""
                <div class="card orange">
                    <h3>🟠 TOTAL</h3>
                    <h1>{total_detected}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------
            # UPDATE CHART
            # -------------------------------
            new_data = pd.DataFrame(
                [[in_count, out_count, live]],
                columns=["IN", "OUT", "LIVE"]
            )

            chart.add_rows(new_data)

            # -------------------------------
            # STORE REPORT
            # -------------------------------
            report_data.append({
                "Time": datetime.now(),
                "IN": in_count,
                "OUT": out_count,
                "LIVE": live,
                "TOTAL": total_detected
            })

        # -------------------------------
        # VIDEO FINISHED
        # -------------------------------
        st.success("✅ Video Processing Completed")

    except Exception as e:

        st.error(f"❌ Error: {e}")

# -------------------------------
# DOWNLOAD REPORT
# -------------------------------
if len(report_data) > 0:

    df = pd.DataFrame(report_data)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="AI_Surveillance_Report.csv",
        mime="text/csv"
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")

st.info("🤖 Powered by YOLOv8 + DeepSORT + Streamlit")