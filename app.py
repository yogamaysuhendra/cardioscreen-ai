import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# CONFIG & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="CARDIOSCREEN AI - Sistem Skrining Jantung",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# LOAD MODEL & ENCODER (CACHED)
# ==============================================================================
@st.cache_resource
def load_ml_components():
    model = joblib.load('model/model_rf_cardioscreen.pkl')
    le_jk = joblib.load('model/encoder_jk.pkl')
    le_rokok = joblib.load('model/encoder_rokok.pkl')
    return model, le_jk, le_rokok

try:
    model_rf, le_jk, le_rokok = load_ml_components()
    jk_options = le_jk.classes_.tolist()
    rokok_options = le_rokok.classes_.tolist()
except Exception as e:
    st.error(f"Gagal memuat komponen Machine Learning. Pastikan file .pkl sudah lengkap di folder 'model'. Error: {e}")
    st.stop()

# ==============================================================================
# CUSTOM CSS - MODERN HEALTHCARE THEME
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111c33 100%);
        color: #f1f5f9;
    }
    
    .luxury-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #38bdf8;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.45) 100%);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.2);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(180, 83, 9, 0.45) 100%);
        border: 1px solid #f59e0b;
        color: #fcd34d;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(4, 120, 87, 0.45) 100%);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5);
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'page' not in st.session_state:
    st.session_state['page'] = 'input_form'
if 'hasil_prediksi' not in st.session_state:
    st.session_state['hasil_prediksi'] = None

# ==============================================================================
# HALAMAN LOGIN
# ==============================================================================
def show_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("""
        <div class="luxury-card" style="text-align: center;">
            <div style="font-size: 3.5rem; margin-bottom: 10px;">🩺</div>
            <h1 class="main-title">CARDIOSCREEN AI</h1>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 5px;">Sistem Skrining Risiko Penyakit Jantung</p>
            <p style="color: #64748b; font-size: 0.8rem;">Puskesmas Bandung • Kabupaten Serang</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("##### 🔐 Autentikasi Petugas / Admin")
            username_input = st.text_input("Username", placeholder="Masukkan username")
            password_input = st.text_input("Password", type="password", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Masuk ke Sistem", use_container_width=True)
            
            if submit_button:
                if username_input == "admin" and password_input == "admin123":
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username_input
                    st.session_state['page'] = 'input_form'
                    st.rerun()
                else:
                    st.error("Username atau Password salah. Gunakan admin / admin123")

# ==============================================================================
# HALAMAN FORM INPUT DATA (8 FITUR + ADMINISTRASI)
# ==============================================================================
def show_input_page():
    st.markdown("<h1 class='main-title'>🩺 Form Skrining Rekam Medis</h1>", unsafe_allow_html=True)
    st.caption("Masukkan data identitas dan 8 indikator klinis utama untuk diproses oleh model Random Forest.")
    st.write("")

    with st.form("input_data_form"):
        # Section A: Administrasi
        st.markdown("""
        <div class="luxury-card">
            <h4 style="color: #38bdf8; margin-bottom: 5px;">A. Identitas Administrasi Pasien</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Digunakan untuk pencatatan rekam medis lokal puskesmas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_adm1, col_adm2 = st.columns(2)
        with col_adm1:
            no_rm = st.text_input("No. Rekam Medis (RM)", placeholder="Contoh: RM-2026-001")
        with col_adm2:
            nama_pasien = st.text_input("Nama Lengkap Pasien", placeholder="Contoh: Budi Santoso")

        st.write("")

        # Section B: Variabel ML (8 Fitur)
        st.markdown("""
        <div class="luxury-card">
            <h4 style="color: #818cf8; margin-bottom: 5px;">B. Variabel Indikator Klinis (8 Fitur Machine Learning)</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Parameter pemeriksaan fisik dan klinis pasien.</p>
        </div>
        """, unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            usia = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=50)
            jenis_kelamin = st.selectbox("Jenis Kelamin", options=jk_options)
            sistolik = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
            diastolik = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=30.0, max_value=150.0, value=80.0)

        with col_f2:
            gula_darah = st.number_input("Gula Darah Sewaktu (mg/dL)", min_value=40.0, max_value=600.0, value=110.0)
            kolesterol = st.number_input("Kolesterol Total (mg/dL)", min_value=50.0, max_value=500.0, value=190.0)
            imt = st.number_input("Indeks Massa Tubuh / IMT (kg/m²)", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
            status_merokok = st.selectbox("Status Merokok", options=rokok_options)

        st.markdown("<br>", unsafe_allow_html=True)
        btn_proses = st.form_submit_button("⚡ Jalankan Klasifikasi Risiko AI", use_container_width=True)

        if btn_proses:
            if not no_rm or not nama_pasien:
                st.warning("Mohon lengkapi No. Rekam Medis dan Nama Pasien terlebih dahulu.")
            else:
                try:
                    jk_encoded = le_jk.transform([jenis_kelamin])[0]
                    rokok_encoded = le_rokok.transform([status_merokok])[0]
                except Exception:
                    jk_encoded = 0
                    rokok_encoded = 0

                input_data = np.array([[
                    usia, 
                    jk_encoded, 
                    sistolik, 
                    diastolik, 
                    gula_darah, 
                    kolesterol, 
                    imt, 
                    rokok_encoded
                ]])

                pred_probs = model_rf.predict_proba(input_data)[0]
                classes = model_rf.classes_
                probabilities = {cls: round(prob, 4) for cls, prob in zip(classes, pred_probs)}

                # --- PENYESUAIAN AMBANG BATAS / THRESHOLD TUNING ---
                prob_rendah = probabilities.get('Risiko Rendah', 0.0)
                prob_sedang = probabilities.get('Risiko Sedang', 0.0)
                prob_tinggi = probabilities.get('Risiko Tinggi', 0.0)

                if prob_rendah > 0.35:
                    pred_class = 'Risiko Rendah'
                elif prob_tinggi >= 0.30 or prob_tinggi > prob_sedang:
                    pred_class = 'Risiko Tinggi'
                else:
                    pred_class = 'Risiko Sedang'
                # ---------------------------------------------------

                fi_values = model_rf.feature_importances_
                feature_names = ['Usia', 'Jenis Kelamin', 'Sistole', 'Diastole', 'Gula Darah', 'Kolesterol', 'IMT', 'Status Merokok']
                feature_importance_data = pd.DataFrame({
                    'Variabel': feature_names, 
                    'Tingkat Kepentingan': fi_values
                }).sort_values(by='Tingkat Kepentingan', ascending=True)

                st.session_state['hasil_prediksi'] = {
                    'no_rm': no_rm,
                    'nama_pasien': nama_pasien,
                    'input_raw': {
                        'Usia': usia, 'Jenis Kelamin': jenis_kelamin, 'Sistolik': sistolik,
                        'Diastolik': diastolik, 'Gula Darah': gula_darah,
                        'Kolesterol': kolesterol, 'IMT': imt, 'Status Merokok': status_merokok
                    },
                    'kelas_final': pred_class,
                    'probabilities': probabilities,
                    'feature_importance': feature_importance_data
                }
                
                st.session_state['page'] = 'hasil_page'
                st.rerun()

# ==============================================================================
# HALAMAN HASIL KLASIFIKASI
# ==============================================================================
def show_hasil_page():
    st.markdown("<h1 class='main-title'>📊 Hasil Analisis Klasifikasi Risiko</h1>", unsafe_allow_html=True)
    st.caption("Hasil komputasi kecerdasan buatan berbasis algoritma Random Forest.")
    st.write("")

    hasil = st.session_state['hasil_prediksi']
    if hasil is None:
        st.warning("Belum ada data pasien yang diproses. Silakan kembali ke form input.")
        if st.button("← Kembali ke Form"):
            st.session_state['page'] = 'input_form'
            st.rerun()
        return

    st.markdown(f"""
    <div class="luxury-card" style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="color:#94a3b8; font-size:0.8rem; letter-spacing: 0.5px;">NO. REKAM MEDIS</span><br>
            <strong style="color:#38bdf8; font-size:1.25rem;">{hasil['no_rm']}</strong>
        </div>
        <div style="text-align: right;">
            <span style="color:#94a3b8; font-size:0.8rem; letter-spacing: 0.5px;">NAMA PASIEN</span><br>
            <strong style="color:#f8fafc; font-size:1.25rem;">{hasil['nama_pasien']}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    kelas_final = hasil['kelas_final']
    probs = hasil['probabilities']
    persentase_final = probs.get(kelas_final, 0) * 100

    st.subheader("1. Klasifikasi Kelas Risiko Utama")
    if kelas_final == "Risiko Tinggi":
        st.markdown(f'<div class="risk-high"><h2>⚠️ {kelas_final}</h2><h1>{persentase_final:.2f}%</h1><p style="margin:5px 0 0 0; font-size:0.9rem;">Pasien memerlukan perhatian medis lanjutan & rujukan spesialis.</p></div>', unsafe_allow_html=True)
    elif kelas_final == "Risiko Sedang":
        st.markdown(f'<div class="risk-medium"><h2>⚡ {kelas_final}</h2><h1>{persentase_final:.2f}%</h1><p style="margin:5px 0 0 0; font-size:0.9rem;">Pasien memerlukan edukasi gaya hidup & pemantauan rutin.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low"><h2>✅ {kelas_final}</h2><h1>{persentase_final:.2f}%</h1><p style="margin:5px 0 0 0; font-size:0.9rem;">Kondisi parameter klinis dalam batas normal.</p></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    st.subheader("2. Detail Probabilitas Model")
    cols = st.columns(len(probs))
    for idx, (cls_name, prob_val) in enumerate(probs.items()):
        with cols[idx]:
            st.markdown(f'<div class="metric-card"><p style="color:#94a3b8; margin:0; font-size:0.85rem;">{cls_name}</p><h2 style="color:#f8fafc; margin:5px 0 0 0;">{prob_val*100:.2f}%</h2></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    st.subheader("3. Tingkat Kepentingan Fitur (Feature Importance)")
    st.caption("Kontribusi relatif masing-masing variabel klinis terhadap keputusan model Random Forest.")
    st.bar_chart(data=hasil['feature_importance'], x='Variabel', y='Tingkat Kepentingan', use_container_width=True)

    st.write("<br>", unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("➕ Input Pasien Baru", use_container_width=True):
            st.session_state['hasil_prediksi'] = None
            st.session_state['page'] = 'input_form'
            st.rerun()
    with col_b2:
        if st.button("📋 Cetak / Tinjau Ringkasan", use_container_width=True):
            st.info("Fitur cetak laporan ringkas siap digunakan untuk lampiran dokumen UAT Puskesmas.")

# ==============================================================================
# MAIN ROUTING & SIDEBAR
# ==============================================================================
def main():
    if not st.session_state['logged_in']:
        show_login_page()
    else:
        st.sidebar.markdown("""
        <div style='text-align:center; padding: 10px 0;'>
            <h2 style='color:#38bdf8; margin-bottom:0;'>CARDIOSCREEN</h2>
            <p style='color:#94a3b8; font-size:0.8rem;'>Puskesmas Bandung Serang</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.write("---")
        nav = st.sidebar.radio("Navigasi Menu:", ["Form Input Data", "Hasil Klasifikasi"], index=0 if st.session_state['page'] == 'input_form' else 1)
        st.session_state['page'] = 'input_form' if nav == "Form Input Data" else 'hasil_page'
        
        st.sidebar.write("---")
        st.sidebar.markdown(f"👤 **Petugas:** `{st.session_state['username']}`")
        if st.sidebar.button("🚪 Keluar (Logout)", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['hasil_prediksi'] = None
            st.rerun()

        if st.session_state['page'] == 'input_form':
            show_input_page()
        elif st.session_state['page'] == 'hasil_page':
            show_hasil_page()

if __name__ == "__main__":
    main()