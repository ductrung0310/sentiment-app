import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from underthesea import word_tokenize

st.title("📱 AI Phân tích đánh giá bình luận APP")
st.write("Sentiment + Phân tích nguyên nhân")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

model = load_model()

# =========================
# KEYWORDS
# =========================

positive_keywords = [
    "tốt","ổn","mượt","nhanh","ok","hay",
    "tuyệt vời","rất tốt","hữu ích","tiện",
    "dễ dùng","đẹp","xịn","đỉnh","ổn áp"
]

negative_keywords = [
    "lỗi","bug","lag","chậm","đơ","crash",
    "treo","không vào được","không dùng được",
    "lỗi đăng nhập","lỗi mạng","giật",
    "đứng máy","quá tệ","tệ","chán",
    "rất tệ"
]

contrast_words = ["nhưng", "tuy nhiên", "dù", "mặc dù"]

# =========================
# PHỦ ĐỊNH CHUẨN
# =========================

strong_negative = [
    "không ổn", "không tốt", "không mượt",
    "không dùng được", "không vào được"
]

medium_negative = [
    "chưa ổn", "chưa tốt", "chưa đẹp", "chưa mượt"
]

# =========================
# XỬ LÝ TEXT
# =========================

def split_contrast(text):
    text = text.lower()
    for w in contrast_words:
        if w in text:
            return text.split(w)[-1]
    return text

# =========================
# SCORE RULE
# =========================

def calc_score(text):
    text = text.lower()
    score = 0

    for w in positive_keywords:
        if w in text:
            if f"không {w}" in text or f"chưa {w}" in text:
                continue
            score += 1

    for w in negative_keywords:
        if w in text:
            score -= 1

    return score

# =========================
# DETECT ISSUE
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

    return ", ".join(issues) if issues else None

# =========================
# AI SCORE
# =========================

def ai_score(text):
    result = model(text)[0]["label"]

    if "5" in result:
        return 2
    elif "4" in result:
        return 1
    elif "3" in result:
        return 0
    elif "2" in result:
        return -1
    else:
        return -2

# =========================
# PHÂN TÍCH CHÍNH
# =========================

def analyze(text):

    original = text.lower()

    # ===== FIX PHỦ ĐỊNH =====
    if any(p in original for p in strong_negative):
        return "🤬 VERY NEGATIVE", detect_issue(original)

    if any(p in original for p in medium_negative):
        return "😡 NEGATIVE", detect_issue(original)

    has_contrast = any(w in original for w in contrast_words)
    main_part = split_contrast(original)

    rule_score = calc_score(original)
    ai = ai_score(main_part)

    has_pos = any(w in original for w in positive_keywords)
    has_neg = any(w in original for w in negative_keywords)

    # ưu tiên vế sau nếu có "nhưng"
    if has_contrast:
        final = calc_score(main_part) + ai * 0.5
    else:
        final = rule_score + ai * 0.5

    # giảm cực đoan nếu vừa khen vừa chê
    if has_pos and has_neg:
        if final > 1:
            sentiment = "🙂 POSITIVE"
        elif final < -1:
            sentiment = "😡 NEGATIVE"
        else:
            sentiment = "😐 NEUTRAL"
    else:
        if final >= 2:
            sentiment = "😍 VERY POSITIVE"
        elif final >= 1:
            sentiment = "🙂 POSITIVE"
        elif final > -1:
            sentiment = "😐 NEUTRAL"
        elif final > -2:
            sentiment = "😡 NEGATIVE"
        else:
            sentiment = "🤬 VERY NEGATIVE"

    return sentiment, detect_issue(original)

# =========================
# UI
# =========================

st.header("📌 Phân tích 1 bình luận")

comment = st.text_area("Nhập đánh giá app")

if st.button("Phân tích"):

    if comment.strip() != "":

        sentiment, issue = analyze(comment)

        if "POSITIVE" in sentiment:
            st.success(sentiment)
        elif "NEGATIVE" in sentiment:
            st.error(sentiment)
        else:
            st.info(sentiment)

        if issue:
            st.write(f"🔍 Nguyên nhân: {issue}")

    else:
        st.warning("Vui lòng nhập")

# =========================
# CSV
# =========================

st.header("📊 Phân tích nhiều bình luận")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    if "comment" not in df.columns:
        st.error("File cần cột 'comment'")
    else:

        sentiments = []
        issues = []

        for c in df["comment"]:
            s, i = analyze(str(c))
            sentiments.append(s)
            issues.append(i if i else "")

        df["sentiment"] = sentiments
        df["issue"] = issues

        st.dataframe(df)

        counts = df["sentiment"].value_counts()

        fig = plt.figure()
        counts.plot(kind="bar")
        plt.title("Sentiment")

        st.pyplot(fig)

        words = []

        for c in df["comment"]:
            tokens = word_tokenize(str(c), format="text").split()
            for w in tokens:
                words.append(w.replace("_"," "))

        wc = WordCloud(width=900, height=450).generate(" ".join(words))

        fig2 = plt.figure()
        plt.imshow(wc)
        plt.axis("off")

        st.pyplot(fig2)
