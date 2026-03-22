import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import re

# =========================
# UI HEADER
# =========================
st.set_page_config(layout="centered")

st.markdown("""
<style>
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📱 AI Phân tích đánh giá APP")
st.caption("Sentiment + Phân tích nguyên nhân (Hybrid AI + Dashboard)")

# ✅ THÔNG BÁO THÊM
st.info("⚡ Mặc định dùng Rule-based (nhanh). Bật AI để tăng độ chính xác (có thể chậm lần đầu).")

# =========================
# OPTION
# =========================
use_ai = st.checkbox("🤖 Dùng AI (chậm hơn)", value=False)
max_rows = st.slider("📏 Giới hạn số dòng xử lý", 10, 1000, 200)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_vi():
    return pipeline("sentiment-analysis",
        model="wonrax/phobert-base-vietnamese-sentiment")

model_vi = None

def get_model_vi():
    global model_vi
    if model_vi is None:
        with st.spinner("⏳ Đang load AI model (lần đầu)..."):
            model_vi = load_model_vi()
    return model_vi

# =========================
# TEXT UTILS
# =========================
def is_vietnamese(text):
    return bool(re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩị]", text.lower()))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9à-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_input(text):
    text = text.strip()

    if not text:
        return False, "Vui lòng nhập bình luận"

    if len(re.sub(r"[a-zA-Zà-ỹ]", "", text)) > len(text) * 0.7:
        return False, "Lỗi: Không nhận diện được"

    if not is_vietnamese(text):
        return False, "Lỗi: Không nhận diện được"

    return True, ""

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
    text = clean_text(text)

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
# AI BATCH
# =========================
def ai_predict_batch(texts):
    try:
        model = get_model_vi()
        results = model(texts)

        output = []
        for r in results:
            if "POS" in r["label"]:
                output.append("🙂 POSITIVE")
            else:
                output.append("😡 NEGATIVE")
        return output
    except:
        return [rule_sentiment(t) for t in texts]

# =========================
# HYBRID
# =========================
def final_sentiment(text):
    rule = rule_sentiment(text)

    if not use_ai:
        return rule

    if "NEGATIVE" in rule:
        return rule

    try:
        model = get_model_vi()
        r = model(text)[0]
        return "🙂 POSITIVE" if "POS" in r["label"] else "😡 NEGATIVE"
    except:
        return rule

# =========================
# ISSUE DETECTION (IMPROVED)
# =========================
def detect_issue(text):
    text = text.lower()
    issues = []

    if any(w in text for w in ["lag","chậm","giật","đơ","delay","load lâu"]):
        issues.append("⚡ Hiệu năng kém")
    if any(w in text for w in ["lỗi","bug","crash","treo","văng"]):
        issues.append("🐞 Lỗi hệ thống")
    if any(w in text for w in ["xấu","rối","khó dùng","không đẹp"]):
        issues.append("🎨 UI/UX kém")
    if any(w in text for w in ["thiếu","không có","cần thêm"]):
        issues.append("➕ Thiếu tính năng")

    # ✅ keyword mơ hồ
    if any(w in text for w in ["chán","tệ","không ổn","khó chịu","trải nghiệm kém"]):
        issues.append("😐 Trải nghiệm người dùng kém")

    # ✅ fallback
    if not issues:
        sentiment = rule_sentiment(text)
        if "NEGATIVE" in sentiment:
            return "❓ Trải nghiệm chưa tốt (cần phân tích thêm)"

    return ", ".join(issues) if issues else ""

# =========================
# SINGLE COMMENT
# =========================
st.header("📌 Phân tích 1 bình luận")
comment = st.text_area("Nhập đánh giá app")

if st.button("Phân tích"):
    valid, msg = is_valid_input(comment)

    if not valid:
        st.warning(msg)
    else:
        final = final_sentiment(comment)

        if "POSITIVE" in final:
            st.success(final)
        elif "NEGATIVE" in final:
            st.error(final)
        else:
            st.info(final)

        issue = detect_issue(comment)
        if issue:
            st.write(f"🔍 Nguyên nhân: {issue}")

# =========================
# MULTI COMMENT
# =========================
st.header("📊 Phân tích nhiều bình luận")
file = st.file_uploader("Upload CSV hoặc Excel", type=["csv","xlsx","xls"])

@st.cache_data
def process_dataframe(df, use_ai_flag):
    df = df.copy()
    comments = df["comment"].astype(str).tolist()

    rule_results = [rule_sentiment(c) for c in comments]

    if not use_ai_flag:
        sentiments = rule_results
    else:
        ai_inputs = []
        ai_index = []

        for i, (c, r) in enumerate(zip(comments, rule_results)):
            if "NEGATIVE" not in r:
                ai_inputs.append(c)
                ai_index.append(i)

        sentiments = rule_results.copy()

        if ai_inputs:
            ai_results = ai_predict_batch(ai_inputs)

            for idx, val in zip(ai_index, ai_results):
                sentiments[idx] = val

    issues = [detect_issue(c) for c in comments]

    df["sentiment"] = sentiments
    df["issue"] = issues

    return df

if file:
    try:
        df = pd.read_csv(file, encoding='utf-8-sig') if file.name.endswith(".csv") else pd.read_excel(file)

        if "comment" not in df.columns:
            st.error("❌ File cần cột 'comment'")
        else:
            # ✅ GIỚI HẠN CỨNG
            effective_rows = max_rows
            if use_ai and max_rows > 200:
                st.warning("⚠️ Bật AI: hệ thống tự giới hạn tối đa 200 dòng để đảm bảo tốc độ.")
                effective_rows = 200

            df = df.head(effective_rows)

            if use_ai and len(df) > 100:
                st.warning("⚠️ Bật AI với dữ liệu lớn có thể chậm. Khuyến nghị ≤ 100 dòng.")

            with st.spinner("⏳ Đang phân tích dữ liệu..."):
                df = process_dataframe(df, use_ai)

            st.success(f"✅ Đã xử lý {len(df)} dòng")

            counts = df["sentiment"].value_counts().reindex([
                "😍 VERY POSITIVE",
                "🙂 POSITIVE",
                "😐 NEUTRAL",
                "😡 NEGATIVE",
                "🤬 VERY NEGATIVE"
            ], fill_value=0)

            total = len(df)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("😍", counts[0], f"{counts[0]/total:.1%}")
            col2.metric("🙂", counts[1], f"{counts[1]/total:.1%}")
            col3.metric("😐", counts[2], f"{counts[2]/total:.1%}")
            col4.metric("😡", counts[3], f"{counts[3]/total:.1%}")
            col5.metric("🤬", counts[4], f"{counts[4]/total:.1%}")

            chart_labels = ["VERY POSITIVE","POSITIVE","NEUTRAL","NEGATIVE","VERY NEGATIVE"]

            colA, colB = st.columns(2)

            with colA:
                fig1, ax1 = plt.subplots()
                ax1.pie(counts.values, labels=chart_labels, autopct='%1.1f%%')
                st.pyplot(fig1)

            with colB:
                fig2, ax2 = plt.subplots()
                ax2.bar(chart_labels, counts.values)
                plt.xticks(rotation=30)
                st.pyplot(fig2)

            option = st.selectbox("🔍 Lọc", ["Tất cả","Tích cực","Trung tính","Tiêu cực"])

            filtered = df.copy()
            if option == "Tích cực":
                filtered = df[df["sentiment"].isin(["😍 VERY POSITIVE","🙂 POSITIVE"])]
            elif option == "Trung tính":
                filtered = df[df["sentiment"]=="😐 NEUTRAL"]
            elif option == "Tiêu cực":
                filtered = df[df["sentiment"].isin(["😡 NEGATIVE","🤬 VERY NEGATIVE"])]

            def color(val):
                colors = {
                    "😍 VERY POSITIVE":"#2ca02c",
                    "🙂 POSITIVE":"#98df8a",
                    "😐 NEUTRAL":"#ffbb78",
                    "😡 NEGATIVE":"#ff7f0e",
                    "🤬 VERY NEGATIVE":"#d62728"
                }
                return f"background-color: {colors.get(val)}"

            st.dataframe(filtered.style.applymap(color, subset=["sentiment"]))

    except Exception as e:
        st.error(f"Lỗi: {e}")
