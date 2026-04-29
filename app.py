# ============================================================
# SQL INJECTION DETECTION STREAMLIT APP
# Created by: Johanes Harindrias Kurniawan
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import joblib
from urllib.parse import unquote_plus

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SQL Injection Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.big-title {
    font-size: 34px;
    font-weight: 800;
    color: #F8FAFC;
}

.sub-title {
    font-size: 16px;
    color: #CBD5E1;
}

.metric-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #334155;
}

.safe-box {
    background-color: #064E3B;
    color: #ECFDF5;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #10B981;
    font-size: 20px;
    font-weight: 700;
}

.alert-box {
    background-color: #7F1D1D;
    color: #FEF2F2;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #EF4444;
    font-size: 20px;
    font-weight: 700;
}

.warning-text {
    color: #FCA5A5;
    font-size: 15px;
}

.safe-text {
    color: #86EFAC;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_assets():
    ml_models = joblib.load("ml_models.pkl")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    dl_model = load_model("dl_model.keras")

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    return ml_models, tokenizer, dl_model, config


try:
    ml_models, tokenizer, dl_model, config = load_assets()
    max_len = int(config["max_len"])
    best_threshold = float(config["best_threshold"])
except Exception as e:
    st.error("Model belum berhasil di-load. Pastikan file ml_models.pkl, tokenizer.pkl, dl_model.keras, dan config.pkl ada di folder yang sama dengan app.py.")
    st.exception(e)
    st.stop()


# ============================================================
# RULE-BASED FUNCTIONS
# ============================================================

sql_keywords = [
    "select", "union", "insert", "update", "delete", "drop",
    "alter", "create", "from", "where", "or", "and",
    "sleep", "benchmark", "waitfor", "delay", "exec",
    "information_schema", "database", "table", "tables",
    "having", "cast", "convert", "concat", "extractvalue",
    "updatexml", "substr", "substring", "ascii", "count",
    "md5", "user", "password"
]


def normalize_payload(text):
    text = str(text).lower()

    # Decode URL encoding
    text = unquote_plus(text)

    # Hilangkan komentar SQL obfuscation seperti /**/
    text = re.sub(r"/\*.*?\*/", " ", text)
    text = text.replace("/**/", " ")

    # Hilangkan spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


def count_special_chars(text):
    return len(re.findall(r"[^a-zA-Z0-9\\s]", str(text)))


def count_digits(text):
    return len(re.findall(r"\\d", str(text)))


def count_sql_keywords(text):
    text_lower = normalize_payload(text)
    total = 0

    for kw in sql_keywords:
        if re.search(r"\\b" + re.escape(kw) + r"\\b", text_lower):
            total += 1

    return total


def has_comment_pattern(text):
    raw = str(text).lower()
    normalized = normalize_payload(text)
    patterns = ["--", "#", "/*", "*/", "/**/"]
    return int(any(p in raw for p in patterns) or any(p in normalized for p in patterns))


def has_boolean_pattern(text):
    """
    Mendeteksi pola boolean SQL Injection:
    - OR 1=1
    - OR (1=1)
    - 1) OR (1=1)--
    - AND 2=2
    - OR 'a'='a'
    - OR TRUE=TRUE
    """
    text_lower = normalize_payload(text)

    patterns = [
        # OR/AND angka sama angka, dengan atau tanpa kurung
        r"\b(or|and)\b\s*\(?\s*\d+\s*=\s*\d+\s*\)?",

        # Bentuk khusus: 1) OR (1=1)--
        r"\d+\s*\)?\s*\b(or|and)\b\s*\(?\s*\d+\s*=\s*\d+\s*\)?",

        # OR/AND string sama string, contoh 'a'='a', '1'='1'
        r"\b(or|and)\b\s*\(?\s*['\"][^'\"]+['\"]\s*=\s*['\"][^'\"]+['\"]\s*\)?",

        # TRUE=TRUE
        r"\b(or|and)\b\s*\(?\s*true\s*=\s*true\s*\)?",

        # Pola 1=1 langsung
        r"\(?\s*1\s*=\s*1\s*\)?"
    ]

    return int(any(re.search(pattern, text_lower) for pattern in patterns))


def has_union_select(text):
    text_lower = normalize_payload(text)
    return int("union" in text_lower and "select" in text_lower)


def has_time_pattern(text):
    text_lower = normalize_payload(text)
    return int(
        "sleep" in text_lower or
        "waitfor" in text_lower or
        "benchmark" in text_lower or
        "delay" in text_lower
    )


def has_encoding_pattern(text):
    text_lower = str(text).lower()
    encoding_patterns = [
        "%27", "%22", "%20", "%3d", "%28", "%29",
        "%2d", "%23", "%2f", "%2a", "%3b"
    ]
    return int(any(p in text_lower for p in encoding_patterns))


def has_stack_query_pattern(text):
    text_lower = normalize_payload(text)
    dangerous_keywords = [
        "drop", "delete", "insert", "update", "select", "alter", "create"
    ]
    return int(";" in text_lower and any(kw in text_lower for kw in dangerous_keywords))


def has_error_based_pattern(text):
    text_lower = normalize_payload(text)
    patterns = [
        "extractvalue", "updatexml", "concat",
        "information_schema", "database()", "user()"
    ]
    return int(any(p in text_lower for p in patterns))


def has_auth_bypass_pattern(text):
    text_lower = normalize_payload(text)
    patterns = [
        r"admin\\s*['\\\"]?\\s*--",
        r"admin\\s*['\\\"]?\\s*#",
        r"login\\s*=\\s*admin",
        r"username\\s*=\\s*admin"
    ]
    return int(any(re.search(p, text_lower) for p in patterns))


def rule_based_indicators(payload):
    indicators = {
        "SQL Keyword": count_sql_keywords(payload),
        "Special Characters": count_special_chars(payload),
        "Digits": count_digits(payload),
        "Boolean Pattern": has_boolean_pattern(payload),
        "Comment Pattern": has_comment_pattern(payload),
        "Union Select": has_union_select(payload),
        "Time Based Pattern": has_time_pattern(payload),
        "Encoding Pattern": has_encoding_pattern(payload),
        "Stack Query Pattern": has_stack_query_pattern(payload),
        "Error Based Pattern": has_error_based_pattern(payload),
        "Auth Bypass Pattern": has_auth_bypass_pattern(payload)
    }
    return indicators


def calculate_rule_score(indicators):
    score = 0

    if indicators["SQL Keyword"] >= 1:
        score += 2
    if indicators["Special Characters"] >= 2:
        score += 1
    if indicators["Boolean Pattern"] == 1:
        score += 5
    if indicators["Comment Pattern"] == 1:
        score += 3
    if indicators["Union Select"] == 1:
        score += 5
    if indicators["Time Based Pattern"] == 1:
        score += 5
    if indicators["Encoding Pattern"] == 1:
        score += 3
    if indicators["Stack Query Pattern"] == 1:
        score += 5
    if indicators["Error Based Pattern"] == 1:
        score += 5
    if indicators["Auth Bypass Pattern"] == 1:
        score += 4

    return score


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_with_dl(payload, threshold):
    seq = tokenizer.texts_to_sequences([payload])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = dl_model.predict(pad, verbose=0).ravel()[0]
    pred = 1 if prob >= threshold else 0
    return pred, prob


def predict_with_ml(payload):
    results = {}

    for name, model in ml_models.items():
        pred = model.predict([payload])[0]
        results[name] = int(pred)

    return results


def hybrid_detection(payload, threshold, rule_threshold):
    indicators = rule_based_indicators(payload)
    rule_score = calculate_rule_score(indicators)

    dl_pred, dl_prob = predict_with_dl(payload, threshold)
    ml_results = predict_with_ml(payload)

    ml_vote = sum(ml_results.values())

    rule_detected = (
        indicators["Boolean Pattern"] == 1 or
        indicators["Comment Pattern"] == 1 or
        indicators["Union Select"] == 1 or
        indicators["Time Based Pattern"] == 1 or
        indicators["Encoding Pattern"] == 1 or
        indicators["Stack Query Pattern"] == 1 or
        indicators["Error Based Pattern"] == 1 or
        indicators["Auth Bypass Pattern"] == 1 or
        rule_score >= rule_threshold
    )

    ml_detected = ml_vote >= 2
    dl_detected = dl_pred == 1

    final_detection = 1 if rule_detected or ml_detected or dl_detected else 0

    return {
        "payload": payload,
        "normalized_payload": normalize_payload(payload),
        "indicators": indicators,
        "rule_score": rule_score,
        "rule_detected": rule_detected,
        "ml_results": ml_results,
        "ml_vote": ml_vote,
        "ml_detected": ml_detected,
        "dl_probability": dl_prob,
        "dl_pred": dl_pred,
        "dl_detected": dl_detected,
        "final_detection": final_detection
    }


# ============================================================
# UI HEADER
# ============================================================

st.markdown('<div class="big-title">🛡️ SQL Injection Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hybrid Detection: Rule-Based Indicator + Machine Learning + Deep Learning</div>',
    unsafe_allow_html=True
)

st.caption("Created by Johanes Harindrias Kurniawan")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "Deep Learning Threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(best_threshold),
    step=0.05
)

rule_threshold = st.sidebar.slider(
    "Rule-Based Threshold",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

st.sidebar.info(
    "Semakin rendah threshold, sistem semakin sensitif terhadap SQL Injection, tetapi False Positive bisa meningkat."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Payload Detection",
    "📂 Batch Detection CSV/Excel",
    "📘 About"
])


# ============================================================
# TAB 1 - SINGLE PAYLOAD
# ============================================================

with tab1:
    st.subheader("Single Payload Detection")

    sample_payloads = {
        "Normal - Search": "search=honda beat",
        "Normal - Product ID": "id=10",
        "SQLi - Boolean": "1) OR (1=1)--",
        "SQLi - Auth Bypass": "admin'--",
        "SQLi - Union": "' UNION SELECT username,password FROM users--",
        "SQLi - Encoded": "%27%20OR%20%281%3D1%29--",
        "SQLi - Time Based": "1; WAITFOR DELAY '0:0:5'--"
    }

    selected_sample = st.selectbox("Pilih contoh payload", list(sample_payloads.keys()))
    default_payload = sample_payloads[selected_sample]

    payload_input = st.text_area(
        "Masukkan Web Request Payload",
        value=default_payload,
        height=120
    )

    if st.button("🚀 Detect Payload", type="primary"):
        result = hybrid_detection(payload_input, threshold, rule_threshold)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rule Score", result["rule_score"])

        with col2:
            st.metric("ML Votes", f'{result["ml_vote"]}/{len(result["ml_results"])}')

        with col3:
            st.metric("DL SQLi Probability", f'{result["dl_probability"] * 100:.2f}%')

        if result["final_detection"] == 1:
            st.markdown(
                '<div class="alert-box">🚨 ALERT WARNING! Payload terdeteksi sebagai SQL Injection.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="warning-text">Rekomendasi: Block request, simpan log payload, investigasi IP/source request, dan gunakan parameterized query.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="safe-box">✅ SAFE. Payload terdeteksi sebagai Normal.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="safe-text">Rekomendasi: Request dapat diproses, namun tetap lakukan monitoring.</div>',
                unsafe_allow_html=True
            )

        st.divider()

        st.subheader("Hybrid Decision Components")

        comp_df = pd.DataFrame({
            "Component": ["Rule-Based", "Machine Learning", "Deep Learning", "Final Decision"],
            "Detected": [
                result["rule_detected"],
                result["ml_detected"],
                result["dl_detected"],
                bool(result["final_detection"])
            ]
        })

        st.dataframe(comp_df, use_container_width=True)

        st.subheader("Rule-Based Indicators")
        indicator_df = pd.DataFrame(
            list(result["indicators"].items()),
            columns=["Indicator", "Value"]
        )
        st.dataframe(indicator_df, use_container_width=True)

        st.subheader("Machine Learning Model Results")
        ml_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Prediction": "SQL Injection" if pred == 1 else "Normal"
                }
                for name, pred in result["ml_results"].items()
            ]
        )
        st.dataframe(ml_df, use_container_width=True)

        st.subheader("Normalized Payload")
        st.code(result["normalized_payload"])


# ============================================================
# TAB 2 - BATCH DETECTION
# ============================================================

with tab2:
    st.subheader("Batch Detection CSV / Excel")

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel. File harus memiliki kolom 'payload'.",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)

        st.write("Preview data:")
        st.dataframe(batch_df.head(), use_container_width=True)

        if "payload" not in batch_df.columns:
            st.error("File wajib memiliki kolom 'payload'.")
        else:
            if st.button("🚀 Run Batch Detection", type="primary"):
                output_rows = []

                progress = st.progress(0)

                for i, row in batch_df.iterrows():
                    payload = str(row["payload"])
                    result = hybrid_detection(payload, threshold, rule_threshold)

                    output_rows.append({
                        "payload": payload,
                        "normalized_payload": result["normalized_payload"],
                        "rule_score": result["rule_score"],
                        "ml_vote": result["ml_vote"],
                        "dl_probability": result["dl_probability"],
                        "rule_detected": result["rule_detected"],
                        "ml_detected": result["ml_detected"],
                        "dl_detected": result["dl_detected"],
                        "final_prediction": "SQL Injection" if result["final_detection"] == 1 else "Normal"
                    })

                    progress.progress((i + 1) / len(batch_df))

                output_df = pd.DataFrame(output_rows)

                st.success("Batch detection selesai.")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "SQL Injection Detected",
                        int((output_df["final_prediction"] == "SQL Injection").sum())
                    )
                with col2:
                    st.metric(
                        "Normal Detected",
                        int((output_df["final_prediction"] == "Normal").sum())
                    )

                st.dataframe(output_df, use_container_width=True)

                csv = output_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Detection Result CSV",
                    data=csv,
                    file_name="streamlit_sql_injection_detection_result.csv",
                    mime="text/csv"
                )


# ============================================================
# TAB 3 - ABOUT
# ============================================================

with tab3:
    st.subheader("About This Application")

    st.write("""
    Aplikasi ini merupakan prototype dashboard deteksi SQL Injection berbasis
    pendekatan hybrid. Sistem menggabungkan tiga komponen utama:
    """)

    st.markdown("""
    1. Rule-Based Indicator untuk mendeteksi pola umum SQL Injection seperti boolean pattern, comment pattern, union select, time-based attack, encoding dan stacked query.
    2. Machine Learning untuk melakukan klasifikasi payload berbasis TF-IDF character n-gram.
    3. Deep Learning untuk membaca pola sequence karakter pada payload.""")

    st.write("""
    Keputusan akhir menggunakan pendekatan hybrid agar sistem lebih sensitif terhadap serangan dan dapat mengurangi risiko False Negative.""")

    st.warning("""
    Prototype ini digunakan untuk kebutuhan pembelajaran dan penelitian.
    Untuk sistem production, tetap diperlukan secure coding, parameterized query, input validation, Web Application Firewall, logging dan monitoring IT Security.
    """)
