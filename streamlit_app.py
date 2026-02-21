import streamlit as st
import requests
import base64
import numpy as np
import cv2

# ================== CONFIG ==================
API_URL = "http://127.0.0.1:8001"
OLLAMA_URL = "http://192.168.51.58:11434"
OLLAMA_MODEL = "qwen2.5:14b"

st.set_page_config(
    page_title="AI Tailor Assistant",
    layout="centered"
)

# ================== SESSION INIT ==================
if "data" not in st.session_state:
    st.session_state.data = None

if "summary" not in st.session_state:
    st.session_state.summary = None


# ================== SAFE JSON ==================
def safe_json(response):
    try:
        return response.json()
    except Exception:
        return {
            "error": "Server mengembalikan response tidak valid",
            "raw_response": response.text
        }


# ================== LOGIC ==================
def infer_gender(measurements):
    chest = measurements.get("chest_circumference", 0)
    shoulder = measurements.get("shoulder_width", 0)

    if shoulder > 42 and chest > 95:
        return "male"
    return "female"


def recommend_size(gender, chest_cm):
    if gender == "male":
        if chest_cm <= 92:
            return "S"
        elif chest_cm <= 98:
            return "M"
        elif chest_cm <= 104:
            return "L"
        elif chest_cm <= 110:
            return "XL"
        elif chest_cm <= 116:
            return "XXL"
        else:
            return "XXXL"
    else:
        if chest_cm <= 84:
            return "S"
        elif chest_cm <= 90:
            return "M"
        elif chest_cm <= 96:
            return "L"
        elif chest_cm <= 102:
            return "XL"
        else:
            return "XXL"


def confidence_label(score: float):
    if score >= 0.85:
        return "🟢 Sangat Tinggi"
    elif score >= 0.7:
        return "🟡 Cukup Baik"
    else:
        return "🔴 Rendah"


def generate_tailor_summary_local(measurements, gender, size, confidence):
    prompt = f"""
Kamu adalah AI Tailor Assistant profesional.

Data hasil pengukuran baju (cm):
{measurements}

Gender (perkiraan): {"Laki-laki" if gender == "male" else "Perempuan"}
Ukuran baju yang direkomendasikan: {size}
Confidence score pengukuran: {confidence}

Tolong:
- Jelaskan arti setiap ukuran (dada, bahu, panjang baju, lengan)
- Jelaskan kenapa ukuran {size} direkomendasikan
- Jelaskan arti confidence score secara singkat
- Fokus pada rekomendasi BAJU (atasan)

Jawaban singkat, jelas, dan profesional.
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.4
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "AI tidak memberikan jawaban.")
    except Exception as e:
        return (
            "⚠️ AI Tailor lokal tidak dapat diakses.\n\n"
            f"Detail error:\n{e}"
        )


# ================== UI ==================
st.title("🧵 AI Tailor Assistant")
st.write("Upload foto tubuh untuk mendapatkan ukuran dan rekomendasi baju")

front = st.file_uploader("📸 Foto Tampak Depan", type=["jpg", "jpeg", "png"])
side = st.file_uploader("📸 Foto Tampak Samping (Opsional)", type=["jpg", "jpeg", "png"])
height = st.number_input("📏 Tinggi badan (cm)", 140, 210, 170)


# ================== BUTTON ACTION ==================
if st.button("🔍 Analyze"):

    if not front:
        st.error("❌ Foto tampak depan wajib diupload.")
        st.stop()

    with st.spinner("🔎 Menganalisis tubuh..."):
        files = {"front": front}
        if side:
            files["left_side"] = side

        try:
            res = requests.post(
                f"{API_URL}/upload_images",
                files=files,
                data={"height_cm": height},
                timeout=120
            )
        except Exception as e:
            st.error("❌ Tidak dapat terhubung ke server AI.")
            st.code(str(e))
            st.stop()

    data = safe_json(res)

    if res.status_code != 200:
        st.error(data.get("error", "Terjadi kesalahan dari server"))
        if "raw_response" in data:
            st.code(data["raw_response"])
        st.stop()

    st.session_state.data = data
    st.session_state.summary = None


# ================== RESULT DISPLAY ==================
data = st.session_state.data

if data:

    measurements = data.get("measurements", {})
    confidence = float(data.get("confidence_score", 0.85))

    st.success("✅ Analisis berhasil!")

    # ================== IMAGE ==================
    if "annotated_image" in data:

        img_bytes = base64.b64decode(data["annotated_image"])
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        st.image(img, channels="BGR", caption="Hasil Deteksi AI", use_container_width=True)

    # ================== CONFIDENCE ==================
    st.subheader("🎯 Confidence Pengukuran")
    st.progress(confidence)
    st.write(f"**Confidence Score:** {confidence} ({confidence_label(confidence)})")

    if confidence < 0.7:
        st.warning(
            "⚠️ Confidence rendah. Disarankan:\n"
            "- Foto full body\n"
            "- Pencahayaan cukup\n"
            "- Jarak kamera lebih jauh"
        )

    # ================== MEASUREMENTS ==================
    st.subheader("📏 Ringkasan Ukuran Baju")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Lingkar Dada", f"{measurements.get('chest_circumference',0)} cm")
        st.metric("Lebar Bahu", f"{measurements.get('shoulder_width',0)} cm")

    with col2:
        st.metric("Panjang Baju", f"{measurements.get('shirt_length',0)} cm")
        st.metric("Panjang Lengan", f"{measurements.get('sleeve_length',0)} cm")

    # ================== SIZE ==================
    gender = infer_gender(measurements)
    size = recommend_size(gender, measurements.get("chest_circumference", 0))

    st.subheader("👤 Profil Tubuh (Perkiraan)")
    st.write(f"Gender: **{'Laki-laki' if gender == 'male' else 'Perempuan'}**")
    st.write(f"Ukuran baju yang disarankan: **{size}**")

    # ================== LLM SUMMARY ==================
    if st.session_state.summary is None:
        with st.spinner("🧠 AI Tailor sedang merangkum hasil..."):
            st.session_state.summary = generate_tailor_summary_local(
                measurements,
                gender,
                size,
                confidence
            )

    st.subheader("🤖 AI Tailor Summary (LLM Lokal)")
    st.write(st.session_state.summary)
