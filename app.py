"""
DEEPFAKE DETECTOR — WEB APP (Streamlit) v2
=============================================
Cara jalankan (di komputer lokal, bukan di Kaggle/Colab):

1. Install dependency (sekali saja):
   pip install streamlit tensorflow opencv-python-headless matplotlib pillow

2. Taruh file model 'deepfake_detector_final.h5' di folder yang sama
   dengan app.py ini (download dari Kaggle Output kalau belum ada).

3. Jalankan:
   streamlit run app.py

4. Browser otomatis terbuka di http://localhost:8501
"""

import streamlit as st
import cv2
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# ==========================================================================
# KONFIGURASI
# ==========================================================================
MODEL_PATH = "deepfake_detector_final.h5"
BEST_THRESHOLD = 0.1846
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fake", "Real"]
LAST_CONV_LAYER = "out_relu"

st.set_page_config(page_title="Deteksi Deepfake", page_icon="🕵️", layout="wide")

# ==========================================================================
# CUSTOM CSS — tema warna & kartu interaktif
# ==========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&display=swap');

    .stApp {
        background-color: #1565c0;
    }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stMarkdown, h1, h2, h3, h4, h5, h6,
    .stTabs button p, .stTabs [data-baseweb="tab"] p,
    .stCheckbox label p, .stFileUploader label, .stFileUploader small,
    .stAlert p, .stCaption, div[data-testid="stCaptionContainer"] p {
        color: #000000;
        font-weight: bold;
    }

    .main-header {
        background: #ffef00;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: #000000;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-family: 'Permanent Marker', cursive;
        font-size: 3.2rem;
        letter-spacing: -1px;
        color: #8b0000;
        margin-bottom: 0.3rem;
        transform: rotate(-1.5deg);
        display: inline-block;
    }
    .main-header p { font-size: 1rem; opacity: 0.85; color: #000000; font-weight: bold; }

    .edu-card {
        background: #ffef00;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 6px solid #2575fc;
        color: #000000;
    }
    .edu-card h4 { margin-top: 0; color: #000000; }
    .edu-card p, .edu-card li, .edu-card ul { color: #000000; }

    .danger-card {
        background: #ffef00;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 6px solid #ff4b4b;
        color: #000000;
    }
    .danger-card h4 { margin-top: 0; color: #000000; }
    .danger-card p, .danger-card li, .danger-card ul { color: #000000; }

    .tip-card {
        background: #ffef00;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 6px solid #2ecc71;
        color: #000000;
    }
    .tip-card h4 { margin-top: 0; color: #000000; }
    .tip-card p, .tip-card li, .tip-card ul { color: #000000; }

    .result-fake {
        background: linear-gradient(135deg, #ff4b4b 0%, #d62828 100%);
        padding: 1.5rem; border-radius: 14px; color: white; text-align: center;
    }
    .result-fake h3, .result-fake p { color: white; }
    .result-real {
        background: linear-gradient(135deg, #2ecc71 0%, #1e8449 100%);
        padding: 1.5rem; border-radius: 14px; color: white; text-align: center;
    }
    .result-real h3, .result-real p { color: white; }
    .result-reject {
        background: linear-gradient(135deg, #f7b733 0%, #fc4a1a 100%);
        padding: 1.5rem; border-radius: 14px; color: white; text-align: center;
    }
    .result-reject h3, .result-reject p { color: white; }
    .badge {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 20px;
        background: rgba(255,255,255,0.25); font-size: 0.85rem; margin-top: 0.5rem;
        color: white;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffef00 !important;
        border-radius: 14px;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #1a1a1a !important;
        border: 1px solid #444 !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }

    /* ---- Y2K chrome accent: tipis, bukan gradient mencolok ---- */
    .main-header, .edu-card, .danger-card, .tip-card,
    [data-testid="stFileUploaderDropzone"] {
        border: 1px solid #c9c9c9;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 2px 8px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    .main-header::before, .edu-card::before, .danger-card::before, .tip-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 40%;
        background: linear-gradient(to bottom, rgba(255,255,255,0.35), rgba(255,255,255,0));
        pointer-events: none;
    }

    /* ---- Ornamen Sherlock Holmes: watermark kaca pembesar, halus ---- */
    .main-header::after {
        content: "🔍";
        position: absolute;
        bottom: -10px;
        right: 10px;
        font-size: 6rem;
        opacity: 0.08;
        pointer-events: none;
    }

    /* ---- Divider bergaya "berkas kasus" ---- */
    .stApp hr {
        border: none;
        border-top: 2px dashed #6b4423;
        margin: 1.2rem 0;
        opacity: 0.6;
    }

    /* ---- Kartu "Case File" untuk teks yang mengambang di atas biru ---- */
    .case-note {
        background: #f2e8d5;
        border: 1px solid #8a6a3e;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
        color: #2b2b2b;
        font-size: 0.9rem;
    }
    .case-note b, .case-note strong { color: #2b2b2b; }
    .case-note-label {
        display: inline-block;
        font-size: 0.7rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #6b4423;
        border: 1px solid #6b4423;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# LOAD MODEL & FACE DETECTOR
# ==========================================================================
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, face_cascade

model, face_cascade = load_assets()

# ==========================================================================
# FUNGSI INTI
# ==========================================================================
def detect_face(image_bgr, min_face_size=(60, 60)):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=min_face_size
    )
    return len(faces) > 0, len(faces)

def preprocess(image_pil):
    img_resized = image_pil.resize(IMG_SIZE)
    arr = np.array(img_resized) / 255.0
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.expand_dims(arr, axis=0), arr

def make_gradcam_heatmap(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(arr, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(heatmap_color * alpha + arr * (1 - alpha), 0, 1)

# --- Fungsi degradasi gambar (untuk robustness test interaktif) ---
def degrade_blur(image_pil, sigma=3):
    arr = np.array(image_pil)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
    return Image.fromarray(blurred)

def degrade_jpeg(image_pil, quality=15):
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def degrade_noise(image_pil, sigma=25):
    arr = np.array(image_pil).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def degrade_resize_updown(image_pil, scale=0.25):
    w, h = image_pil.size
    small = image_pil.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

ROBUSTNESS_CONDITIONS = {
    "Blur": degrade_blur,
    "Kompresi JPEG rendah": degrade_jpeg,
    "Noise": degrade_noise,
    "Resize turun-naik": degrade_resize_updown,
}

def predict_pil(image_pil):
    """Prediksi untuk satu gambar PIL, termasuk cek gatekeeper wajah."""
    image_bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    has_face, n_faces = detect_face(image_bgr)
    if not has_face:
        return {"label": None, "confidence": None, "has_face": False}
    arr_batch, _ = preprocess(image_pil.convert("RGB"))
    prob_real = model.predict(arr_batch, verbose=0)[0][0]
    pred_class = 1 if prob_real > BEST_THRESHOLD else 0
    confidence = prob_real if pred_class == 1 else 1 - prob_real
    return {"label": CLASS_NAMES[pred_class], "confidence": confidence, "has_face": True}

# ==========================================================================
# HEADER
# ==========================================================================
st.markdown("""
<div class="main-header">
    <h1>🕵️ Deteksi Citra Deepfake</h1>
    <p>CNN (MobileNetV2, Transfer Learning) — dilengkapi gatekeeper deteksi wajah & Grad-CAM</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Deteksi Gambar", "📚 Edukasi Deepfake", "ℹ️ Tentang Sistem"])

# ==========================================================================
# TAB 1 — DETEKSI
# ==========================================================================
with tab1:
    col_toggle1, col_toggle2 = st.columns(2)
    with col_toggle1:
        show_gradcam_toggle = st.checkbox("🔥 Tampilkan Grad-CAM (area fokus model)", value=True)
    with col_toggle2:
        show_robustness_toggle = st.checkbox("🧪 Uji Ketahanan (Robustness Test) pada gambar ini", value=False)
        st.markdown("""
        <div class="case-note">
            <span class="case-note-label">📋 Catatan Kasus</span><br>
            Menguji apakah prediksi tetap konsisten saat gambar diburamkan,
            dikompresi, diberi noise, atau di-resize — mensimulasikan kondisi
            gambar dunia nyata. Lihat tab <b>Tentang Sistem</b> untuk penjelasan lengkap.
        </div>
        """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Pilih gambar wajah (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        has_face, n_faces = detect_face(image_np_bgr)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption="Gambar yang diupload", use_column_width=True)

        if not has_face:
            with col2:
                st.markdown("""
                <div class="result-reject">
                    <h3>🚫 DITOLAK</h3>
                    <p>Tidak terdeteksi wajah manusia pada gambar ini.</p>
                    <span class="badge">Sistem ini khusus analisis wajah</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            arr_batch, arr = preprocess(image_pil)
            prob_real = model.predict(arr_batch, verbose=0)[0][0]
            pred_class = 1 if prob_real > BEST_THRESHOLD else 0
            pred_label = CLASS_NAMES[pred_class]
            confidence = prob_real if pred_class == 1 else 1 - prob_real

            with col2:
                css_class = "result-real" if pred_label == "Real" else "result-fake"
                icon = "✅" if pred_label == "Real" else "⚠️"
                st.markdown(f"""
                <div class="{css_class}">
                    <h3>{icon} Prediksi: {pred_label}</h3>
                    <p style="font-size:1.8rem; font-weight:bold; margin:0.3rem 0;">{confidence:.1%}</p>
                    <span class="badge">{n_faces} wajah terdeteksi</span>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 Detail probabilitas mentah"):
                    st.write(f"Probabilitas kelas Real: `{prob_real:.4f}`")
                    st.write(f"Threshold keputusan: `{BEST_THRESHOLD}`")

            if show_gradcam_toggle:
                st.markdown("---")
                heatmap = make_gradcam_heatmap(arr_batch)
                overlay = overlay_gradcam(arr, heatmap)
                colA, colB = st.columns(2)
                with colA:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True, clamp=True)
                with colB:
                    st.image(overlay, caption="Overlay — area fokus model", use_column_width=True)
                st.info(
                    "💡 Area **merah/kuning** menunjukkan bagian gambar yang paling "
                    "memengaruhi keputusan model. Bandingkan dengan area wajah "
                    "yang biasanya rentan artefak deepfake (garis rambut, mata, mulut)."
                )

            if show_robustness_toggle:
                st.markdown("---")
                st.subheader("🧪 Hasil Uji Ketahanan (Robustness Test)")
                st.caption(
                    "Gambar yang sama diuji ulang setelah 'dirusak' dengan beberapa "
                    "cara, untuk melihat apakah prediksi model tetap konsisten pada "
                    "kondisi gambar dunia nyata (screenshot, kompresi media sosial, dll)."
                )

                cols = st.columns(len(ROBUSTNESS_CONDITIONS) + 1)

                with cols[0]:
                    st.image(image_pil, caption="Asli (bersih)", use_column_width=True)
                    badge_color = "#2ecc71" if pred_label == "Real" else "#ff4b4b"
                    st.markdown(
                        f"<p style='text-align:center; color:{badge_color}; font-weight:bold;'>"
                        f"{pred_label} ({confidence:.1%})</p>",
                        unsafe_allow_html=True
                    )

                for i, (cond_name, degrade_fn) in enumerate(ROBUSTNESS_CONDITIONS.items(), start=1):
                    degraded_img = degrade_fn(image_pil)
                    result = predict_pil(degraded_img)
                    with cols[i]:
                        st.image(degraded_img, caption=cond_name, use_column_width=True)
                        if not result["has_face"]:
                            st.markdown(
                                "<p style='text-align:center; color:#f7b733; font-weight:bold;'>"
                                "Wajah tidak terdeteksi</p>",
                                unsafe_allow_html=True
                            )
                        else:
                            badge_color = "#2ecc71" if result["label"] == "Real" else "#ff4b4b"
                            same_as_original = result["label"] == pred_label
                            change_icon = "✓" if same_as_original else "⚠️"
                            st.markdown(
                                f"<p style='text-align:center; color:{badge_color}; font-weight:bold;'>"
                                f"{change_icon} {result['label']} ({result['confidence']:.1%})</p>",
                                unsafe_allow_html=True
                            )

                n_changed = sum(
                    1 for cond_name, degrade_fn in ROBUSTNESS_CONDITIONS.items()
                    if predict_pil(degrade_fn(image_pil))["label"] != pred_label
                )
                if n_changed == 0:
                    st.success("✅ Prediksi **konsisten** di semua kondisi degradasi yang diuji.")
                else:
                    st.warning(
                        f"⚠️ Prediksi **berubah** pada {n_changed} dari "
                        f"{len(ROBUSTNESS_CONDITIONS)} kondisi degradasi — model belum "
                        "sepenuhnya tahan terhadap gambar berkualitas rendah."
                    )

    st.markdown("---")

# ==========================================================================
# TAB 2 — EDUKASI
# ==========================================================================
with tab2:
    st.markdown("""
    <div class="edu-card">
        <h4>🎭 Apa itu Deepfake?</h4>
        <p><b>Deepfake</b> adalah citra, video, atau audio sintetis yang dihasilkan menggunakan
        teknik kecerdasan buatan (khususnya <i>deep learning</i>, seperti Generative Adversarial
        Networks/GAN atau Autoencoder) untuk menukar atau memanipulasi wajah dan suara seseorang
        secara sangat meyakinkan — seolah-olah orang tersebut benar-benar mengatakan atau
        melakukan sesuatu yang sebenarnya tidak pernah terjadi.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="edu-card">
            <h4>⚙️ Bagaimana Deepfake Dibuat?</h4>
            <p>Secara umum, teknik pembuatan deepfake melibatkan:</p>
            <ul>
                <li><b>Face-swapping:</b> menukar wajah seseorang dengan wajah target
                menggunakan model encoder-decoder</li>
                <li><b>Face reenactment:</b> memanipulasi ekspresi/gerakan wajah target
                agar mengikuti gerakan wajah sumber</li>
                <li><b>Voice cloning:</b> mensintesis suara seseorang berdasarkan sampel
                audio aslinya</li>
            </ul>
            <p>Teknologi ini membutuhkan banyak data citra/video wajah target untuk
            melatih model agar hasilnya realistis.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="danger-card">
            <h4>⚠️ Dampak & Risiko</h4>
            <ul>
                <li><b>Disinformasi politik</b> — video palsu tokoh publik</li>
                <li><b>Penipuan finansial</b> — impersonasi suara/wajah CEO dalam video call</li>
                <li><b>Pencemaran nama baik</b> — konten memalukan yang tidak pernah terjadi</li>
                <li><b>Erosi kepercayaan publik</b> — sulit membedakan mana bukti asli</li>
            </ul>
            <p>Karena itu, riset deteksi deepfake menjadi penting sebagai
            langkah mitigasi teknologi ini.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-card">
        <h4>🔎 Tanda-Tanda Umum Deepfake (Secara Visual)</h4>
        <ul>
            <li>Transisi tidak natural di garis rambut, telinga, atau leher</li>
            <li>Kedipan mata yang tidak wajar (terlalu jarang/tidak sinkron)</li>
            <li>Pencahayaan wajah tidak konsisten dengan latar belakang</li>
            <li>Tekstur kulit terlalu halus atau blur di area tertentu</li>
            <li>Gerakan bibir tidak sinkron dengan audio (khusus video)</li>
        </ul>
        <p><i>Catatan: seiring perkembangan teknologi, tanda-tanda ini semakin sulit
        dikenali mata telanjang — itulah mengapa deteksi berbasis AI seperti pada
        sistem ini menjadi relevan.</i></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="edu-card">
        <h4>🧠 Bagaimana CNN Mendeteksi Deepfake?</h4>
        <p>Model <i>Convolutional Neural Network</i> (CNN) belajar mengenali pola tekstur,
        artefak kompresi, dan inkonsistensi statistik piksel yang biasanya muncul akibat
        proses generasi/manipulasi AI — pola yang seringkali tidak kasat mata bagi manusia,
        namun dapat dipelajari dari ribuan contoh citra asli dan citra hasil manipulasi.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================================
# TAB 3 — TENTANG SISTEM
# ==========================================================================
with tab3:
    st.markdown("""
    <div class="edu-card">
        <h4>🏗️ Arsitektur Model</h4>
        <p>Sistem ini menggunakan <b>MobileNetV2</b> yang dilatih ulang (transfer learning,
        dua tahap: feature extraction + fine-tuning) pada dataset citra wajah Fake/Real.</p>
    </div>
    <div class="edu-card">
        <h4>🛡️ Gatekeeper Deteksi Wajah</h4>
        <p>Sebelum diklasifikasi, gambar diperiksa dulu menggunakan Haar Cascade untuk
        memastikan mengandung wajah manusia — mencegah gambar acak/tidak relevan
        dipaksa diklasifikasi Fake/Real.</p>
    </div>
    <div class="edu-card">
        <h4>🎯 Threshold Optimal</h4>
        <p>Ambang keputusan (0,1846) dipilih berdasarkan <b>Youden's J Statistic</b>
        dari kurva ROC data uji, untuk menyeimbangkan performa antar kelas.</p>
    </div>
    <div class="edu-card">
        <h4>🧪 Apa itu Uji Ketahanan (Robustness Test)?</h4>
        <p>Ini adalah pengujian untuk menjawab pertanyaan: <i>"seberapa tetap akurat
        model ini kalau gambarnya tidak sempurna?"</i></p>
        <p>Di dunia nyata, gambar yang beredar jarang sebersih dataset pelatihan —
        misalnya karena di-screenshot, dikompresi ulang saat dikirim lewat media
        sosial, atau kualitas kamera yang kurang baik. Kalau model hanya diuji pada
        gambar bersih, belum tentu model tersebut benar-benar berguna pada kondisi
        gambar dunia nyata.</p>
        <p>Fitur ini menguji gambar yang Anda upload dalam 4 kondisi degradasi:</p>
        <ul>
            <li><b>Blur</b> — gambar diburamkan, mensimulasikan foto tidak fokus</li>
            <li><b>Kompresi JPEG rendah</b> — mensimulasikan gambar yang berkali-kali
            di-screenshot atau dikirim ulang lewat aplikasi chat/media sosial</li>
            <li><b>Noise</b> — bintik-bintik acak ditambahkan, mensimulasikan kualitas
            kamera/kondisi pencahayaan buruk</li>
            <li><b>Resize turun-naik</b> — gambar diperkecil lalu diperbesar lagi,
            mensimulasikan hilangnya detail akibat kompresi ukuran file</li>
        </ul>
        <p>Sistem lalu menunjukkan apakah prediksi model <b>tetap konsisten</b> pada
        tiap kondisi tersebut, atau justru <b>berubah</b> — yang mengindikasikan
        model masih rentan terhadap penurunan kualitas gambar. Pengujian jenis ini
        merupakan salah satu isu yang menjadi perhatian dalam riset deteksi
        deepfake terkini, mengingat metode deteksi yang tidak tahan terhadap
        degradasi gambar berisiko tidak efektif digunakan pada kondisi nyata.</p>
    </div>
    """, unsafe_allow_html=True)
