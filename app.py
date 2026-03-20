import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

st.title("📱 AI Phân tích đánh giá APP")
st.write("Sentiment + Phân tích nguyên nhân (Nhanh hơn, hỗ trợ CSV/Excel)")

# =========================
# OPTION
# =========================
use_ai = st.checkbox("🤖 Dùng AI (chậm hơn)", value=False)
max_rows = st.slider("📏 Giới hạn số dòng xử lý", 10, 1000, 200)

# =========================
# LOAD MODEL (LAZY LOAD)
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = None

def get_model():
    global model
    if model is None:
        with st.spinner("⏳ Đang load model AI lần đầu..."):
            model = load_model()
    return model

# =========================
# KEYWORDS
# =========================
positive_keywords = {
    "tốt": 2, "ổn": 1.5, "mượt": 2, "nhanh": 1.5,
    "ok": 1, "hay": 2, "tuyệt vời": 3,
    "rất tốt": 3, "hữu ích": 2, "tiện": 1.5,
    "dễ dùng": 2, "đẹp": 2, "xịn": 2, "đỉnh": 2.5,
    "ổn áp": 1.5, "thích": 2.5
}

negative_keywords = {
    "lỗi": -3, "bug": -3, "lag": -2.5, "chậm": -2,
    "đơ": -2, "crash": -3, "treo": -2.5,
    "không vào được": -3, "không dùng được": -3,
    "lỗi đăng nhập": -3, "giật": -2,
    "quá tệ": -3, "tệ": -2, "chán": -1.5,
    "rất tệ": -3
}

strong_negative = [
    "không ổn", "không tốt", "không mượt",
    "không dùng được", "không vào được"
]

medium_negative = [
    "chưa ổn", "chưa tốt", "chưa đẹp", "chưa mượt"
]

# =========================
# RULE-BASED
# =========================
def calc_score(text):
    score = 0
    text = text.lower()

    for phrase in strong_negative:
        if phrase in text:
            score -= 3

    for phrase in medium_negative:
        if phrase in text:
            score -= 2

    for w, val in positive_keywords.items():
        if w in text:
            if f"không {w}" in text or f"chưa {w}" in text:
                score -= abs(val)
            else:
                score += val

    for w, val in negative_keywords.items():
        if w in text:
            score += val

    return score

def rule_sentiment(text):
    score = calc_score(text)

    if score >= 2:
        return "😍 VERY POSITIVE"
    elif score >= 1:
        return "🙂 POSITIVE"
    elif score > -1:
        return "😐 NEUTRAL"
    elif score > -2.5:
        return "😡 NEGATIVE"
    else:
        return "🤬 VERY NEGATIVE"

# =========================
# AI SENTIMENT (BATCH)
# =========================
def ai_predict_batch(texts):
    model = get_model()
    results = model(texts, batch_size=16)

    scores = []
    for r in results:
        if r["label"] == "POSITIVE":
            scores.append(1)
        else:
            scores.append(-1)
    return scores

# =========================
# ISSUE DETECTION
# =========================
def detect_issue(text):
    text = text.lower()
    issues = []

    if any(w in text for w in ["lag","chậm","giật","đơ"]):
        issues.append("⚡ Hiệu năng kém")
    if any(w in text for w in ["lỗi","bug","crash","treo"]):
        issues.append("🐞 Lỗi hệ thống")
    if any(w in text for w in ["xấu","rối","khó dùng","không đẹp","chưa đẹp"]):
        issues.append("🎨 UI/UX kém")
    if any(w in text for w in ["thiếu","không có","cần thêm"]):
        issues.append("➕ Thiếu tính năng")

    return ", ".join(issues) if issues else ""

# =========================
# SINGLE COMMENT
# =========================
st.header("📌 Phân tích 1 bình luận")
comment = st.text_area("Nhập đánh giá app")

if st.button("Phân tích"):
    if comment.strip():
        rule = rule_sentiment(comment)

        if use_ai:
            ai_val = ai_predict_batch([comment])[0]
            final = "🙂 POSITIVE" if ai_val == 1 else "😡 NEGATIVE"
        else:
            final = rule

        if "POSITIVE" in final:
            st.success(final)
        elif "NEGATIVE" in final:
            st.error(final)
        else:
            st.info(final)

        issue = detect_issue(comment)
        if issue:
            st.write(f"🔍 Nguyên nhân: {issue}")
    else:
        st.warning("Vui lòng nhập")

# =========================
# MULTI COMMENT
# =========================
st.header("📊 Phân tích nhiều bình luận")
file = st.file_uploader("Upload CSV hoặc Excel", type=["csv","xlsx","xls"])

@st.cache_data
def process_dataframe(df, use_ai):
    df = df.copy()
    comments = df["comment"].astype(str).tolist()

    sentiments = [rule_sentiment(c) for c in comments]

    if use_ai:
        ai_scores = ai_predict_batch(comments)
        sentiments = [
            "🙂 POSITIVE" if s == 1 else "😡 NEGATIVE"
            for s in ai_scores
        ]

    issues = [detect_issue(c) for c in comments]

    df["sentiment"] = sentiments
    df["issue"] = issues

    return df

if file:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding='utf-8-sig')
        else:
            df = pd.read_excel(file)

        if "comment" not in df.columns:
            st.error("❌ File cần cột 'comment'")
        else:
            df = df.head(max_rows)

            with st.spinner("⏳ Đang phân tích dữ liệu..."):
                df = process_dataframe(df, use_ai)

            st.success(f"✅ Đã xử lý {len(df)} dòng")

            # =========================
            # THỐNG KÊ
            # =========================
            st.subheader("📊 Thống kê kết quả")

            counts = df["sentiment"].value_counts().reindex([
                "😍 VERY POSITIVE",
                "🙂 POSITIVE",
                "😐 NEUTRAL",
                "😡 NEGATIVE",
                "🤬 VERY NEGATIVE"
            ], fill_value=0)

            total = len(df)

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("😍 Rất tích cực", counts["😍 VERY POSITIVE"], f"{counts['😍 VERY POSITIVE']/total:.1%}")
            col2.metric("🙂 Tích cực", counts["🙂 POSITIVE"], f"{counts['🙂 POSITIVE']/total:.1%}")
            col3.metric("😐 Trung lập", counts["😐 NEUTRAL"], f"{counts['😐 NEUTRAL']/total:.1%}")
            col4.metric("😡 Tiêu cực", counts["😡 NEGATIVE"], f"{counts['😡 NEGATIVE']/total:.1%}")
            col5.metric("🤬 Rất tiêu cực", counts["🤬 VERY NEGATIVE"], f"{counts['🤬 VERY NEGATIVE']/total:.1%}")

            # =========================
            # FILTER
            # =========================
            st.subheader("🔍 Lọc bình luận")

            filter_option = st.selectbox(
                "Chọn loại bình luận",
                ["Tất cả", "Tích cực", "Trung tính", "Tiêu cực"]
            )

            filtered_df = df.copy()

            if filter_option == "Tích cực":
                filtered_df = df[df["sentiment"].isin(["😍 VERY POSITIVE", "🙂 POSITIVE"])]

            elif filter_option == "Trung tính":
                filtered_df = df[df["sentiment"] == "😐 NEUTRAL"]

            elif filter_option == "Tiêu cực":
                filtered_df = df[df["sentiment"].isin(["😡 NEGATIVE", "🤬 VERY NEGATIVE"])]
            st.write(f"📌 Hiển thị {len(filtered_df)} / {len(df)} bình luận")

            # ===== TABLE COLOR =====
            def color_sentiment(val):
                colors = {
                    "😍 VERY POSITIVE":"#2ca02c",
                    "🙂 POSITIVE":"#98df8a",
                    "😐 NEUTRAL":"#ffbb78",
                    "😡 NEGATIVE":"#ff7f0e",
                    "🤬 VERY NEGATIVE":"#d62728"
                }
                return f"background-color: {colors.get(val,'white')}"

            st.dataframe(filtered_df.style.applymap(color_sentiment, subset=["sentiment"]))

            # ===== BAR CHART (THEO FILTER) =====
            counts_filtered = filtered_df["sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.bar(counts_filtered.index, counts_filtered.values)
            ax.set_title("📊 Biểu đồ sentiment")
            plt.xticks(rotation=30)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi: {e}")
