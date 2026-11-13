import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime
from fpdf import FPDF
from supabase import create_client, Client

# ==============================================================================
# CONFIG & PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="CARDIOSCREEN AI — Digital Heart Screening",
    page_icon=":material/cardiology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# BACKEND CLOUD DATABASE: SUPABASE (100% PERMANEN DI CLOUD)
# ==============================================================================
@st.cache_resource
def init_supabase() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_supabase()

if supabase is None:
    st.error(
        "Konfigurasi Cloud Database (Supabase) belum ditemukan. "
        "Pastikan `SUPABASE_URL` dan `SUPABASE_KEY` sudah diisi pada `.streamlit/secrets.toml` (lokal) atau menu Secrets di Streamlit Community Cloud.",
        icon=":material/database:"
    )
    st.stop()

def simpan_skrining_db(data_dict):
    supabase.table("riwayat_skrining").insert(data_dict).execute()

def ambil_semua_riwayat_db():
    response = supabase.table("riwayat_skrining").select("*").order("id", desc=True).execute()
    rows = response.data if response.data else []
    
    hasil_list = []
    for row in rows:
        hasil_list.append({
            'ID': row['id'],
            'Waktu': row['waktu'],
            'No. RM': row['no_rm'],
            'Nama Pasien': row['nama_pasien'],
            'Usia': row['usia'],
            'JK': row['jenis_kelamin'],
            'Tensi': f"{int(row['sistolik'])}/{int(row['diastolik'])}",
            'Gula Darah': f"{row['gula_darah']} mg/dL",
            'Kolesterol': f"{row['kolesterol']} mg/dL",
            'IMT': row['imt'],
            'Merokok': row['status_merokok'],
            'Hasil Risiko': row['hasil_risiko'],
            'Keyakinan Model': row['keyakinan_model'],
            'Petugas': row['petugas']
        })
    return hasil_list

def hapus_riwayat_by_id(record_id):
    supabase.table("riwayat_skrining").delete().eq("id", record_id).execute()

# ==============================================================================
# LOAD MODEL & ENCODER (CACHED - 100% PRESERVED)
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
    st.error(f"Gagal memuat komponen Machine Learning. Pastikan file .pkl lengkap di folder 'model'. Error: {e}")
    st.stop()

# ==============================================================================
# HELPER FUNCTIONS: GENERATE EXCEL (.XLSX) & PDF (.PDF)
# ==============================================================================
def generate_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Rekap Skrining')
    return output.getvalue()

def generate_pdf_bytes(hasil):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header Kop Puskesmas
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(67, 56, 202)
    pdf.cell(0, 8, "UPT PUSKESMAS BANDUNG - KABUPATEN SERANG", ln=True, align="C")
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "LAPORAN HASIL SKRINING AWAL RISIKO PENYAKIT JANTUNG (CARDIOSCREEN AI)", ln=True, align="C")
    
    pdf.set_draw_color(99, 102, 241)
    pdf.set_line_width(0.6)
    pdf.line(10, 28, 200, 28)
    pdf.ln(8)

    # Identitas Pasien
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(238, 242, 255)
    pdf.cell(0, 8, "  A. IDENTITAS PASIEN", ln=True, fill=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    identitas = [
        ("No. Rekam Medis", hasil['no_rm']),
        ("Nama Lengkap Pasien", hasil['nama_pasien']),
        ("Waktu Pemeriksaan", hasil['waktu'])
    ]
    for label, val in identitas:
        pdf.cell(55, 7, label, border=0)
        pdf.cell(5, 7, ":", border=0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, str(val), ln=True)
        pdf.set_font("Helvetica", "", 10)
    pdf.ln(4)

    # Parameter Klinis
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "  B. PARAMETER INDIKATOR KLINIS", ln=True, fill=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    for param, val in hasil['input_raw'].items():
        clean_val = str(val).replace("²", "2")
        pdf.cell(55, 7, param, border=0)
        pdf.cell(5, 7, ":", border=0)
        pdf.cell(0, 7, clean_val, ln=True)
    pdf.ln(4)

    # Hasil Analisis AI
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "  C. HASIL KLASIFIKASI ALGORITMA RANDOM FOREST", ln=True, fill=True)
    pdf.ln(3)

    kelas = hasil['kelas_final']
    persen = hasil['probabilities'].get(kelas, 0) * 100

    pdf.set_font("Helvetica", "B", 12)
    if kelas == "Risiko Tinggi":
        pdf.set_text_color(225, 29, 72)
    elif kelas == "Risiko Sedang":
        pdf.set_text_color(217, 119, 6)
    else:
        pdf.set_text_color(5, 150, 105)

    pdf.cell(0, 9, f"Kategori Risiko: {kelas.upper()} (Keyakinan Model: {persen:.2f}%)", ln=True)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Rincian Probabilitas Semua Kelas:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for cls_name, prob_val in hasil['probabilities'].items():
        pdf.cell(0, 6, f"- {cls_name}: {prob_val*100:.2f}%", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 5, "Catatan Medis: Dokumen ini merupakan hasil skrining awal berbasis kecerdasan buatan (CARDIOSCREEN AI) dan digunakan sebagai alat bantu pendukung keputusan klinis. Diagnosis akhir tetap memerlukan pemeriksaan langsung oleh dokter.")

    return bytes(pdf.output())

# ==============================================================================
# SVG ICONS & LUMINOUS CARDIO-TECH BRAND ASSETS (FLUSH-LEFT)
# ==============================================================================
BRAND_LOGO_SVG = '<svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="56" height="56" rx="16" fill="url(#cardio_tech_grad)"/><rect x="1.5" y="1.5" width="53" height="53" rx="14.5" stroke="white" stroke-opacity="0.35" stroke-width="1.5"/><path d="M28 42C28 42 13.5 33.5 13.5 22.5C13.5 18.1 17 14.6 21.3 14.6C24 14.6 26.4 16.1 28 18.3C29.6 16.1 32 14.6 34.7 14.6C39 14.6 42.5 18.1 42.5 22.5C42.5 33.5 28 42 28 42Z" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M18 25.5H22.8L25 20.5L28.5 30.2L30.8 24.2H38" stroke="#FFE4E6" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/><defs><linearGradient id="cardio_tech_grad" x1="0" y1="0" x2="56" y2="56" gradientUnits="userSpaceOnUse"><stop stop-color="#2563EB"/><stop offset="0.55" stop-color="#6366F1"/><stop offset="1" stop-color="#F43F5E"/></linearGradient></defs></svg>'

ECG_DIVIDER_SVG = '<svg width="100%" height="26" viewBox="0 0 400 26" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0 13H152L166 5L180 21L195 2L210 24L224 13H400" stroke="url(#ecg_cardio_grad)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/><defs><linearGradient id="ecg_cardio_grad" x1="0" y1="13" x2="400" y2="13" gradientUnits="userSpaceOnUse"><stop stop-color="#CBD5E1" stop-opacity="0.15"/><stop offset="0.38" stop-color="#4F46E5"/><stop offset="0.52" stop-color="#F43F5E"/><stop offset="0.66" stop-color="#0EA5E9"/><stop offset="1" stop-color="#CBD5E1" stop-opacity="0.15"/></linearGradient></defs></svg>'

ICON_USERS = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
ICON_ALERT = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>'
ICON_WARN = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
ICON_CHECK = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
ICON_CLIPBOARD = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M9 12h6"/><path d="M9 16h6"/></svg>'
ICON_ACTIVITY = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
ICON_HISTORY = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>'

# ==============================================================================
# CUSTOM CSS - VIBRANT MESH + DOT-GRID CANVAS, LAYERED SHADOWS & MOBILE LIGHT LOCK
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root, .stApp {
    color-scheme: light !important;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0F172A;
}

/* Rich Multi-Tone Mesh Gradient + Subtle Medical Dot-Grid Pattern on ALL App Layers */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #EEF2FF !important;
    background-image: 
        radial-gradient(rgba(79, 70, 229, 0.14) 1px, transparent 1px),
        radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.18) 0px, transparent 52%),
        radial-gradient(at 100% 0%, rgba(244, 63, 94, 0.14) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(14, 165, 233, 0.16) 0px, transparent 55%),
        radial-gradient(at 0% 100%, rgba(168, 85, 247, 0.12) 0px, transparent 50%) !important;
    background-size: 24px 24px, 100% 100%, 100% 100%, 100% 100%, 100% 100% !important;
    background-attachment: fixed !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* PERJELAS TOMBOL NAVIGASI SIDEBAR (>>) DI POJOK KIRI ATAS HP */
[data-testid="stSidebarCollapsedControl"] button,
button[kind="header"],
[data-testid="collapsedControl"] button {
    background-color: #FFFFFF !important;
    color: #1E1B4B !important;
    border: 1.5px solid #A5B4FC !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.22) !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
button[kind="header"] svg,
[data-testid="collapsedControl"] svg {
    fill: #1E1B4B !important;
    color: #1E1B4B !important;
    stroke: #1E1B4B !important;
}

/* PAKSA SEMUA JUDUL, SUBHEADER & LABEL TETAP GELAP DI DARK MODE HP */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
[data-testid="stHeadingWithActionElements"] *,
label, label p {
    color: #0F172A !important;
}

[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
}

/* Elevated Multi-Layer Shadow Glass Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255, 255, 255, 0.94) !important;
    backdrop-filter: blur(14px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.9) !important;
    border-radius: 22px !important;
    box-shadow: 
        0 4px 10px -2px rgba(15, 23, 42, 0.05),
        0 20px 40px -8px rgba(79, 70, 229, 0.13) !important;
    transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1) !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #A5B4FC !important;
    box-shadow: 
        0 10px 20px -3px rgba(244, 63, 94, 0.10),
        0 26px 48px -8px rgba(79, 70, 229, 0.18) !important;
}

/* Clean Modern Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(248,250,252,0.97) 100%) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1.5px solid #E2E8F0 !important;
    box-shadow: 6px 0 30px rgba(79, 70, 229, 0.06) !important;
}

[data-testid="stSidebarContent"] {
    padding-top: 1.2rem !important;
}

/* SINGLE-BORDER INPUTS WITH INDIGO-CORAL INTERACTIVE FOCUS RING & MOBILE HIGH CONTRAST */
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background-color: #F8FAFC !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #F8FAFC !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 13px !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.22s cubic-bezier(0.22, 1, 0.36, 1) !important;
}

div[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #6366F1 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08) !important;
}

div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: #4F46E5 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.18) !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: #F8FAFC !important;
    -webkit-text-fill-color: #0F172A !important;
    color: #0F172A !important;
    caret-color: #4F46E5 !important;
    border: none !important;
    box-shadow: none !important;
    font-weight: 600 !important;
    padding-left: 14px !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder {
    -webkit-text-fill-color: #64748B !important;
    color: #64748B !important;
    opacity: 1 !important;
}

div[data-testid="stTextInput"] button svg {
    fill: #475569 !important;
    color: #475569 !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

div[data-testid="stNumberInput"] button {
    background-color: transparent !important;
    border: none !important;
    color: #4338CA !important;
    font-weight: 700 !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: #EEF2FF !important;
    color: #3730A3 !important;
}
div[data-testid="stNumberInput"] button:active {
    transform: scale(0.92) !important;
}

/* Pulsing Status Dot */
@keyframes pulse-live {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.9); }
    70% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: #34D399;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse-live 1.8s infinite;
}

/* Luminous Hero Banner & Sub-Page Headers */
.hero-gradient-box, .welcome-banner {
    background: linear-gradient(125deg, #2563EB 0%, #4F46E5 55%, #E11D48 100%) !important;
    border-radius: 22px;
    box-shadow: 
        0 10px 24px -5px rgba(79, 70, 229, 0.34),
        0 22px 44px -10px rgba(225, 29, 72, 0.24);
    border: 1px solid rgba(255, 255, 255, 0.28);
}

.welcome-banner {
    padding: 28px 32px;
    margin-bottom: 24px;
}

.subpage-banner {
    background: linear-gradient(125deg, #1D4ED8 0%, #4F46E5 60%, #E11D48 100%) !important;
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 
        0 10px 22px -5px rgba(79, 70, 229, 0.30),
        0 18px 36px -8px rgba(225, 29, 72, 0.20);
    border: 1px solid rgba(255, 255, 255, 0.28);
}

.hero-gradient-box *, .welcome-banner *, .subpage-banner * {
    color: #FFFFFF !important;
}

/* Interactive Step Pills inside Form Banner */
.step-pill {
    background: rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.30);
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    transition: transform 0.2s ease, background 0.2s ease;
}
.step-pill:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.26);
}

/* Interactive Clinical Reference Chips */
.ref-chip {
    background: rgba(255, 255, 255, 0.92);
    border: 1.5px solid #E0E7FF;
    border-radius: 15px;
    padding: 12px 16px;
    box-shadow: 0 6px 16px -4px rgba(79, 70, 229, 0.10);
    transition: all 0.25s cubic-bezier(0.22, 1, 0.36, 1);
}
.ref-chip:hover {
    transform: translateY(-3px);
    border-color: #6366F1;
    box-shadow: 0 12px 24px -4px rgba(79, 70, 229, 0.18);
}

/* Section Header Strip inside Form Cards */
.section-strip {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 10px;
    margin-bottom: 14px;
    border-bottom: 1.5px dashed #E2E8F0;
}

.section-badge-indigo {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    color: #FFFFFF;
    font-size: 0.78rem;
    font-weight: 800;
    padding: 4px 11px;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
}

.section-badge-coral {
    background: linear-gradient(135deg, #4F46E5 0%, #E11D48 100%);
    color: #FFFFFF;
    font-size: 0.78rem;
    font-weight: 800;
    padding: 4px 11px;
    border-radius: 8px;
    box-shadow: 0 4px 10px rgba(225, 29, 72, 0.25);
}

/* Patient Header Card */
.care-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #EEF2FF 100%);
    border: 1.5px solid #C7D2FE;
    border-left: 6px solid #4F46E5;
    border-radius: 18px;
    padding: 22px 26px;
    box-shadow: 0 12px 28px -6px rgba(79, 70, 229, 0.12);
    margin-bottom: 20px;
}

/* Interactive Metric Card with Multi-Tone Elevation */
.stat-card {
    background-color: rgba(255, 255, 255, 0.96);
    border: 1.5px solid #E2E8F0;
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 
        0 4px 8px -2px rgba(15, 23, 42, 0.04),
        0 14px 28px -6px rgba(79, 70, 229, 0.10);
    display: flex;
    align-items: center;
    gap: 15px;
    transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1);
    cursor: default;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 12px 20px -4px rgba(244, 63, 94, 0.14),
        0 22px 38px -6px rgba(79, 70, 229, 0.18);
    border-color: #818CF8;
}

.icon-box-grad {
    width: 50px;
    height: 50px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 8px 16px -3px rgba(79, 70, 229, 0.28);
}

.stat-label {
    color: #475569;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 3px;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.1;
}

/* High-Contrast Clinical Risk Banners */
.risk-high {
    background: linear-gradient(145deg, #FFF1F2 0%, #FFE4E6 100%);
    border: 2px solid #F43F5E;
    color: #881337;
    padding: 28px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 16px 32px -6px rgba(244, 63, 94, 0.24);
}

.risk-medium {
    background: linear-gradient(145deg, #FFFBEB 0%, #FEF3C7 100%);
    border: 2px solid #F59E0B;
    color: #78350F;
    padding: 28px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 16px 32px -6px rgba(245, 158, 11, 0.24);
}

.risk-low {
    background: linear-gradient(145deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 2px solid #10B981;
    color: #064E3B;
    padding: 28px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 16px 32px -6px rgba(16, 185, 129, 0.24);
}

/* VIBRANT INTERACTIVE PRIMARY BUTTON (COBALT -> INDIGO -> CARDIO CORAL) */
.stButton > button[kind="primary"], .stFormSubmitButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 55%, #E11D48 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
    padding: 12px 22px !important;
    box-shadow: 
        0 6px 14px -2px rgba(79, 70, 229, 0.38),
        0 12px 24px -4px rgba(225, 29, 72, 0.24) !important;
    transition: all 0.24s cubic-bezier(0.22, 1, 0.36, 1) !important;
}

.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #4338CA 50%, #F43F5E 100%) !important;
    transform: translateY(-2.5px) !important;
    box-shadow: 
        0 10px 20px -2px rgba(79, 70, 229, 0.44),
        0 18px 32px -4px rgba(244, 63, 94, 0.32) !important;
}

.stButton > button[kind="primary"]:active, .stFormSubmitButton > button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.28) !important;
}

/* INTERACTIVE SECONDARY & DOWNLOAD BUTTONS */
.stButton > button[kind="secondary"], .stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    box-shadow: 0 4px 10px -2px rgba(15, 23, 42, 0.06) !important;
    transition: all 0.22s cubic-bezier(0.22, 1, 0.36, 1) !important;
}

.stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #EEF2FF 0%, #FFF1F2 100%) !important;
    border-color: #6366F1 !important;
    color: #4338CA !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 18px -3px rgba(99, 102, 241, 0.20) !important;
}

.stButton > button[kind="secondary"]:active, .stDownloadButton > button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* Sidebar Navigation Hover Slide */
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    transform: translateX(5px) !important;
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
if 'nav_menu' not in st.session_state:
    st.session_state['nav_menu'] = "Dashboard Utama"
if 'hasil_prediksi' not in st.session_state:
    st.session_state['hasil_prediksi'] = None

# ==============================================================================
# 1. HALAMAN LOGIN (CUSTOM CREDENTIALS DARI ST.SECRETS)
# ==============================================================================
def show_login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.1, 1.35, 1.1])
    
    with col2:
        with st.container(border=True):
            login_brand_html = (
                f'<div class="hero-gradient-box" style="padding: 28px 22px; text-align: center; margin-bottom: 14px;">'
                f'<div style="display: flex; justify-content: center; margin-bottom: 14px;">{BRAND_LOGO_SVG}</div>'
                f'<div style="font-weight: 800; margin: 0; font-size: 1.65rem; letter-spacing: -0.3px; text-shadow: 0 2px 8px rgba(0,0,0,0.2);">CARDIOSCREEN AI</div>'
                f'<div style="font-size: 0.9rem; margin: 6px 0 14px 0; font-weight: 500; opacity: 0.95;">Sistem Skrining Dini Risiko Penyakit Jantung</div>'
                f'<span style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); font-size: 0.75rem; font-weight: 700; padding: 5px 14px; border-radius: 20px; display: inline-block;">UPT Puskesmas Bandung</span>'
                f'</div>'
                f'{ECG_DIVIDER_SVG}'
            )
            st.markdown(login_brand_html, unsafe_allow_html=True)

            with st.form("login_form", border=False):
                username_input = st.text_input("Username Petugas", placeholder="Masukkan username petugas")
                password_input = st.text_input("Kata Sandi", type="password", placeholder="Masukkan kata sandi")
                st.write("")
                submit_button = st.form_submit_button(
                    "Masuk ke Portal Skrining",
                    icon=":material/login:",
                    use_container_width=True
                )
                
                if submit_button:
                    valid_user = st.secrets.get("ADMIN_USERNAME", "")
                    valid_pass = st.secrets.get("ADMIN_PASSWORD", "")

                    if not valid_user or not valid_pass:
                        st.error("Konfigurasi akun admin belum diatur pada Secrets.", icon=":material/warning:")
                    elif username_input.strip() == valid_user and password_input == valid_pass:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username_input.strip()
                        st.session_state['nav_menu'] = "Dashboard Utama"
                        st.rerun()
                    else:
                        st.error("Username atau kata sandi tidak sesuai. Silakan periksa kembali.", icon=":material/error:")

# ==============================================================================
# 2. DASHBOARD UTAMA (MEMBACA REAL-TIME DARI CLOUD DATABASE)
# ==============================================================================
def show_dashboard_page():
    riwayat = ambil_semua_riwayat_db()
    total_pasien = len(riwayat)
    jml_tinggi = sum(1 for r in riwayat if r['Hasil Risiko'] == 'Risiko Tinggi')
    jml_sedang = sum(1 for r in riwayat if r['Hasil Risiko'] == 'Risiko Sedang')
    jml_rendah = sum(1 for r in riwayat if r['Hasil Risiko'] == 'Risiko Rendah')

    banner_html = (
        f'<div class="welcome-banner">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">'
        f'<div>'
        f'<div style="display:inline-flex; align-items:center; gap:6px; background: rgba(255,255,255,0.18); padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.4px;"><span style="width:7px; height:7px; background:#FFE4E6; border-radius:50%; display:inline-block;"></span>PORTAL PETUGAS KESEHATAN • {st.session_state["username"].upper()}</div>'
        f'<h1 style="margin: 12px 0 8px 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.4px;">Dashboard Skrining CARDIOSCREEN AI</h1>'
        f'<p style="margin: 0; font-size: 0.94rem; opacity: 0.95; max-width: 640px;">Sistem pendukung keputusan klinis berbasis algoritma Random Forest untuk deteksi dini risiko penyakit jantung di UPT Puskesmas Bandung.</p>'
        f'</div>'
        f'<div style="background: rgba(255, 255, 255, 0.16); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.32); padding: 15px 22px; border-radius: 16px;">'
        f'<div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px; opacity: 0.9;">STATUS ENGINE & CLOUD DB</div>'
        f'<div style="font-size: 0.95rem; font-weight: 700; margin-top: 5px; display: flex; align-items: center;"><span class="status-dot"></span>RF & Supabase Aktif</div>'
        f'</div></div></div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    st.subheader("Ringkasan Total Pemeriksaan (Cloud Database)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card" style="border-top: 4px solid #4F46E5;"><div class="icon-box-grad" style="background: linear-gradient(135deg, #2563EB 0%, #6366F1 100%);">{ICON_USERS}</div><div><div class="stat-label">TOTAL DIPERIKSA</div><div class="stat-value" style="color: #312E81;">{total_pasien}</div></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card" style="border-top: 4px solid #F43F5E;"><div class="icon-box-grad" style="background: linear-gradient(135deg, #E11D48 0%, #FB7185 100%);">{ICON_ALERT}</div><div><div class="stat-label">RISIKO TINGGI</div><div class="stat-value" style="color: #E11D48;">{jml_tinggi}</div></div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card" style="border-top: 4px solid #F59E0B;"><div class="icon-box-grad" style="background: linear-gradient(135deg, #D97706 0%, #FBBF24 100%);">{ICON_WARN}</div><div><div class="stat-label">RISIKO SEDANG</div><div class="stat-value" style="color: #D97706;">{jml_sedang}</div></div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card" style="border-top: 4px solid #10B981;"><div class="icon-box-grad" style="background: linear-gradient(135deg, #059669 0%, #34D399 100%);">{ICON_CHECK}</div><div><div class="stat-label">RISIKO RENDAH</div><div class="stat-value" style="color: #059669;">{jml_rendah}</div></div></div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("Akses Cepat Layanan")
    m1, m2, m3 = st.columns(3)

    with m1:
        with st.container(border=True):
            st.markdown("##### Skrining Pasien Baru")
            st.caption("Input nomor rekam medis serta 8 parameter klinis untuk analisis risiko jantung.")
            if st.button("Mulai Skrining", key="btn_dash_input", icon=":material/add_circle:", type="primary", use_container_width=True):
                st.session_state['nav_menu'] = "Form Input Data"
                st.rerun()

    with m2:
        with st.container(border=True):
            st.markdown("##### Riwayat Pemeriksaan")
            st.caption("Tinjau seluruh riwayat pasien yang tersimpan permanen di Cloud dan ekspor Excel.")
            if st.button("Buka Riwayat", key="btn_dash_riwayat", icon=":material/history:", type="secondary", use_container_width=True):
                st.session_state['nav_menu'] = "Riwayat Skrining"
                st.rerun()

    with m3:
        with st.container(border=True):
            st.markdown("##### Hasil Analisis Terakhir")
            st.caption("Lihat kembali detail klasifikasi, interpretasi pasien, dan unduh laporan PDF.")
            if st.button("Lihat Hasil", key="btn_dash_hasil", icon=":material/analytics:", type="secondary", use_container_width=True):
                st.session_state['nav_menu'] = "Hasil Klasifikasi"
                st.rerun()

    st.write("")
    with st.expander("Panduan Alur Penggunaan untuk Petugas Puskesmas", icon=":material/menu_book:", expanded=True):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("**1. Siapkan Data Rekam Medis**")
            st.caption("Pastikan data usia, tekanan darah (sistolik/diastolik), gula darah sewaktu, kolesterol total, IMT, dan riwayat merokok tersedia.")
        with col_g2:
            st.markdown("**2. Isi Form & Gunakan Bantuan (?)**")
            st.caption("Pilih **Mulai Skrining**. Arahkan kursor ke ikon tanda tanya (?) di setiap kolom input untuk melihat nilai rujukan normal.")
        with col_g3:
            st.markdown("**3. Tersimpan Otomatis di Cloud**")
            st.caption("Setiap hasil skrining otomatis tersimpan permanen ke Cloud Database dan dapat diunduh sebagai PDF maupun rekap Excel.")

# ==============================================================================
# 3. HALAMAN FORM INPUT DATA (LOGIKA ML 100% VOTING STANDAR + SIMPAN CLOUD DB)
# ==============================================================================
def show_input_page():
    input_banner_html = (
        f'<div class="subpage-banner">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">'
        f'<div style="display: flex; align-items: center; gap: 16px;">'
        f'<div style="background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35); width: 52px; height: 52px; border-radius: 15px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">{ICON_CLIPBOARD}</div>'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 800; letter-spacing: 0.6px; opacity: 0.9;">MODUL INPUT REKAM MEDIS</div>'
        f'<h1 style="margin: 2px 0 4px 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.4px;">Form Skrining Risiko Jantung</h1>'
        f'<p style="margin: 0; font-size: 0.88rem; opacity: 0.92;">Lengkapi identitas administrasi serta 8 indikator klinis untuk analisis Random Forest.</p>'
        f'</div></div>'
        f'<div style="display: flex; gap: 8px; flex-wrap: wrap;">'
        f'<div class="step-pill">1. Identitas RM</div>'
        f'<div class="step-pill">2. 8 Indikator Klinis</div>'
        f'<div class="step-pill">3. Simpan Cloud AI</div>'
        f'</div></div></div>'
    )
    st.markdown(input_banner_html, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown('<div class="ref-chip" style="border-left: 4px solid #2563EB;"><div style="font-size:0.7rem; font-weight:800; color:#475569;">RUJUKAN TENSI NORMAL</div><div style="font-size:0.95rem; font-weight:800; color:#1E3A8A; margin-top:2px;">90–120 / 60–80 mmHg</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="ref-chip" style="border-left: 4px solid #4F46E5;"><div style="font-size:0.7rem; font-weight:800; color:#475569;">GULA DARAH SEWAKTU</div><div style="font-size:0.95rem; font-weight:800; color:#312E81; margin-top:2px;">&lt; 140 mg/dL</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="ref-chip" style="border-left: 4px solid #E11D48;"><div style="font-size:0.7rem; font-weight:800; color:#475569;">KOLESTEROL TOTAL</div><div style="font-size:0.95rem; font-weight:800; color:#9F1239; margin-top:2px;">&lt; 200 mg/dL</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown('<div class="ref-chip" style="border-left: 4px solid #10B981;"><div style="font-size:0.7rem; font-weight:800; color:#475569;">INDEKS MASSA TUBUH</div><div style="font-size:0.95rem; font-weight:800; color:#065F46; margin-top:2px;">18,5 – 24,9 kg/m²</div></div>', unsafe_allow_html=True)

    st.write("")

    with st.form("input_data_form", border=False):
        with st.container(border=True):
            st.markdown(
                '<div class="section-strip">'
                '<span class="section-badge-indigo">BAGIAN A</span>'
                '<span style="font-size: 1.08rem; font-weight: 800; color: #0F172A;">Identitas Administrasi Pasien</span>'
                '</div>',
                unsafe_allow_html=True
            )
            col_adm1, col_adm2 = st.columns(2)
            with col_adm1:
                no_rm = st.text_input("No. Rekam Medis (RM)", placeholder="Contoh: RM-2026-001", help="Nomor registrasi rekam medis pasien.")
            with col_adm2:
                nama_pasien = st.text_input("Nama Lengkap Pasien", placeholder="Contoh: Budi Santoso", help="Nama lengkap pasien sesuai identitas.")

        with st.container(border=True):
            st.markdown(
                '<div class="section-strip">'
                '<span class="section-badge-coral">BAGIAN B</span>'
                '<span style="font-size: 1.08rem; font-weight: 800; color: #0F172A;">Indikator Pemeriksaan Klinis (8 Parameter)</span>'
                '</div>',
                unsafe_allow_html=True
            )
            st.caption("Arahkan kursor ke ikon (?) di sebelah label kolom untuk melihat rentang nilai normal masing-masing parameter.")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                usia = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=50, help="Usia pasien dalam tahun.")
                jenis_kelamin = st.selectbox("Jenis Kelamin", options=jk_options, help="Jenis kelamin pasien.")
                sistolik = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=50.0, max_value=250.0, value=120.0, help="Angka atas tekanan darah. Normal: 90–120 mmHg.")
                diastolik = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=30.0, max_value=150.0, value=80.0, help="Angka bawah tekanan darah. Normal: 60–80 mmHg.")

            with col_f2:
                gula_darah = st.number_input("Gula Darah Sewaktu (mg/dL)", min_value=40.0, max_value=600.0, value=110.0, help="Kadar glukosa darah sewaktu. Normal umum: < 140 mg/dL.")
                kolesterol = st.number_input("Kolesterol Total (mg/dL)", min_value=50.0, max_value=500.0, value=190.0, help="Kadar kolesterol total. Normal yang disarankan: < 200 mg/dL.")
                imt = st.number_input("Indeks Massa Tubuh / IMT (kg/m²)", min_value=10.0, max_value=60.0, value=22.5, step=0.1, help="Normal: 18,5 – 24,9 kg/m².")
                status_merokok = st.selectbox("Status Merokok", options=rokok_options, help="Kebiasaan merokok pasien.")

        st.write("")
        btn_proses = st.form_submit_button(
            "Jalankan Klasifikasi Risiko AI",
            icon=":material/bolt:",
            use_container_width=True
        )

        if btn_proses:
            if not no_rm.strip() or not nama_pasien.strip():
                st.warning("Mohon lengkapi **No. Rekam Medis** dan **Nama Lengkap Pasien** terlebih dahulu.", icon=":material/warning:")
            else:
                # --- LOGIKA MACHINE LEARNING MURNI SESUAI DATASET & SKRIPSI ---
                try:
                    jk_encoded = le_jk.transform([jenis_kelamin])[0]
                    rokok_encoded = le_rokok.transform([status_merokok])[0]
                except Exception:
                    jk_encoded = 0
                    rokok_encoded = 0

                input_data = np.array([[
                    usia, jk_encoded, sistolik, diastolik, 
                    gula_darah, kolesterol, imt, rokok_encoded
                ]])

                pred_probs = model_rf.predict_proba(input_data)[0]
                classes = model_rf.classes_
                probabilities = {cls: round(prob, 4) for cls, prob in zip(classes, pred_probs)}

                # Keputusan final menggunakan voting standar Random Forest (probabilitas tertinggi)
                pred_class = max(probabilities, key=probabilities.get)
                # --------------------------------------------------------------

                fi_values = model_rf.feature_importances_
                feature_names = ['Usia', 'Jenis Kelamin', 'Sistole', 'Diastole', 'Gula Darah', 'Kolesterol', 'IMT', 'Status Merokok']
                feature_importance_data = pd.DataFrame({
                    'Variabel': feature_names, 
                    'Tingkat Kepentingan': fi_values
                }).sort_values(by='Tingkat Kepentingan', ascending=True)

                waktu_periksa = datetime.now().strftime("%d-%m-%Y %H:%M")
                keyakinan_str = f"{probabilities.get(pred_class, 0)*100:.1f}%"

                # Simpan permanen ke Supabase Cloud Database
                try:
                    simpan_skrining_db({
                        'waktu': waktu_periksa,
                        'no_rm': no_rm.strip(),
                        'nama_pasien': nama_pasien.strip(),
                        'usia': int(usia),
                        'jenis_kelamin': jenis_kelamin,
                        'sistolik': float(sistolik),
                        'diastolik': float(diastolik),
                        'gula_darah': float(gula_darah),
                        'kolesterol': float(kolesterol),
                        'imt': float(imt),
                        'status_merokok': status_merokok,
                        'hasil_risiko': pred_class,
                        'keyakinan_model': keyakinan_str,
                        'petugas': st.session_state['username']
                    })
                except Exception as db_err:
                    st.error(f"Klasifikasi berhasil, namun gagal menyimpan ke Cloud Database: {db_err}")
                    st.stop()

                st.session_state['hasil_prediksi'] = {
                    'waktu': waktu_periksa,
                    'no_rm': no_rm.strip(),
                    'nama_pasien': nama_pasien.strip(),
                    'input_raw': {
                        'Usia': f"{usia} thn", 'Jenis Kelamin': jenis_kelamin,
                        'Tekanan Darah': f"{int(sistolik)}/{int(diastolik)} mmHg",
                        'Gula Darah': f"{gula_darah} mg/dL", 'Kolesterol': f"{kolesterol} mg/dL",
                        'IMT': f"{imt} kg/m²", 'Merokok': status_merokok
                    },
                    'kelas_final': pred_class,
                    'probabilities': probabilities,
                    'feature_importance': feature_importance_data
                }
                
                st.session_state['nav_menu'] = "Hasil Klasifikasi"
                st.rerun()

# ==============================================================================
# 4. HALAMAN HASIL KLASIFIKASI (DENGAN TABEL & GRAFIK FEATURE IMPORTANCE)
# ==============================================================================
def show_hasil_page():
    header_html = (
        f'<div class="subpage-banner">'
        f'<div style="display: flex; align-items: center; gap: 16px;">'
        f'<div style="background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35); width: 52px; height: 52px; border-radius: 15px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">{ICON_ACTIVITY}</div>'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 800; letter-spacing: 0.6px; opacity: 0.9;">OUTPUT ANALISIS RANDOM FOREST</div>'
        f'<h1 style="margin: 2px 0 4px 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.4px;">Hasil Analisis Skrining Jantung</h1>'
        f'<p style="margin: 0; font-size: 0.88rem; opacity: 0.92;">Ringkasan klasifikasi risiko, interpretasi bahasa pasien, dan rekomendasi tindak lanjut klinis.</p>'
        f'</div></div></div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    hasil = st.session_state['hasil_prediksi']
    if hasil is None:
        with st.container(border=True):
            st.info("Belum ada data pasien yang diproses pada sesi ini.", icon=":material/info:")
            if st.button("Ke Form Input Data", icon=":material/arrow_back:", type="primary"):
                st.session_state['nav_menu'] = "Form Input Data"
                st.rerun()
        return

    patient_card_html = (
        f'<div class="care-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">'
        f'<div><span style="color:#64748B; font-size:0.75rem; font-weight:800; letter-spacing:0.5px;">NO. REKAM MEDIS</span><br><strong style="color:#4F46E5; font-size:1.2rem;">{hasil["no_rm"]}</strong></div>'
        f'<div><span style="color:#64748B; font-size:0.75rem; font-weight:800; letter-spacing:0.5px;">NAMA PASIEN</span><br><strong style="color:#0F172A; font-size:1.2rem;">{hasil["nama_pasien"]}</strong></div>'
        f'<div style="text-align: right;"><span style="color:#64748B; font-size:0.75rem; font-weight:800; letter-spacing:0.5px;">WAKTU SKRINING</span><br><span style="color:#1E293B; font-size:0.95rem; font-weight:700;">{hasil["waktu"]}</span></div>'
        f'</div>'
    )
    st.markdown(patient_card_html, unsafe_allow_html=True)

    kelas_final = hasil['kelas_final']
    probs = hasil['probabilities']
    persentase_final = probs.get(kelas_final, 0) * 100

    st.subheader("1. Keputusan Klasifikasi & Interpretasi Pasien")
    col_res1, col_res2 = st.columns([1.1, 1.4])

    with col_res1:
        if kelas_final == "Risiko Tinggi":
            st.markdown(f'<div class="risk-high"><div style="display:inline-flex; align-items:center; gap:6px; background:#F43F5E; color:#FFFFFF; padding:5px 14px; border-radius:20px; font-size:0.78rem; font-weight:800; letter-spacing:0.4px; box-shadow: 0 4px 10px rgba(244,63,94,0.3);">STATUS PRIORITAS</div><h2 style="margin:12px 0 4px 0; color:#881337; font-weight:800;">{kelas_final.upper()}</h2><h1 style="margin:4px 0; font-size:2.7rem; color:#E11D48; font-weight:800;">{persentase_final:.1f}%</h1><p style="margin:6px 0 0 0; font-size:0.9rem; font-weight:600;">Perlu perhatian medis lanjutan & rujukan dokter.</p></div>', unsafe_allow_html=True)
        elif kelas_final == "Risiko Sedang":
            st.markdown(f'<div class="risk-medium"><div style="display:inline-flex; align-items:center; gap:6px; background:#F59E0B; color:#FFFFFF; padding:5px 14px; border-radius:20px; font-size:0.78rem; font-weight:800; letter-spacing:0.4px; box-shadow: 0 4px 10px rgba(245,158,11,0.3);">STATUS WASPADA</div><h2 style="margin:12px 0 4px 0; color:#78350F; font-weight:800;">{kelas_final.upper()}</h2><h1 style="margin:4px 0; font-size:2.7rem; color:#D97706; font-weight:800;">{persentase_final:.1f}%</h1><p style="margin:6px 0 0 0; font-size:0.9rem; font-weight:600;">Waspada, memerlukan pemantauan klinis rutin.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><div style="display:inline-flex; align-items:center; gap:6px; background:#10B981; color:#FFFFFF; padding:5px 14px; border-radius:20px; font-size:0.78rem; font-weight:800; letter-spacing:0.4px; box-shadow: 0 4px 10px rgba(16,185,129,0.3);">STATUS TERKENDALI</div><h2 style="margin:12px 0 4px 0; color:#064E3B; font-weight:800;">{kelas_final.upper()}</h2><h1 style="margin:4px 0; font-size:2.7rem; color:#059669; font-weight:800;">{persentase_final:.1f}%</h1><p style="margin:6px 0 0 0; font-size:0.9rem; font-weight:600;">Indikator klinis saat ini berada pada batas aman.</p></div>', unsafe_allow_html=True)

    with col_res2:
        with st.container(border=True):
            st.markdown("##### Panduan Penjelasan ke Pasien / Lansia")
            if kelas_final == "Risiko Tinggi":
                st.write(
                    f"**Arti Hasil:** Kondisi jantung Bapak/Ibu **{hasil['nama_pasien']}** berada pada kategori **perlu perhatian serius**. "
                    "Terdapat kombinasi faktor klinis yang berisiko memicu gangguan jantung."
                )
                st.markdown("**Tindak Lanjut & Edukasi Kesehatan:**\n- Segera jadwalkan pemeriksaan dokter atau rujukan EKG.\n- Batasi garam (< 1 sendok teh/hari), gorengan, dan santan.\n- Hindari aktivitas fisik berat secara mendadak.")
            elif kelas_final == "Risiko Sedang":
                st.write(
                    f"**Arti Hasil:** Kondisi Bapak/Ibu **{hasil['nama_pasien']}** tergolong **waspada (risiko menengah)**. "
                    "Jika pola hidup tidak diperbaiki, tingkat risiko dapat meningkat."
                )
                st.markdown("**Tindak Lanjut & Edukasi Kesehatan:**\n- Kontrol rutin tekanan darah, gula darah, dan kolesterol tiap bulan.\n- Perbanyak sayur, buah, serta kurangi rokok dan lemak.\n- Jalan santai rutin 30 menit, 3–5 kali seminggu.")
            else:
                st.write(
                    f"**Arti Hasil:** Indikator kesehatan Bapak/Ibu **{hasil['nama_pasien']}** saat ini **tergolong aman (risiko rendah)**. "
                    "Tetap pertahankan pola hidup sehat agar risiko tidak meningkat."
                )
                st.markdown("**Tindak Lanjut & Edukasi Kesehatan:**\n- Pertahankan berat badan ideal dan hindari paparan asap rokok.\n- Terapkan pola makan gizi seimbang.\n- Lakukan skrining ulang berkala setiap 6 bulan.")

    st.write("")
    st.subheader("2. Rincian Probabilitas Model")
    cols = st.columns(len(probs))
    border_colors = ["#10B981", "#F59E0B", "#F43F5E"]
    for idx, (cls_name, prob_val) in enumerate(probs.items()):
        top_col = border_colors[idx % len(border_colors)]
        with cols[idx]:
            st.markdown(f'<div class="stat-card" style="justify-content: center; text-align: center; border-top: 4px solid {top_col};"><div><div class="stat-label">{cls_name.upper()}</div><div class="stat-value" style="color: #0F172A; font-size: 1.65rem;">{prob_val*100:.2f}%</div></div></div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("Ringkasan Data Klinis & Grafik Faktor Penentu", icon=":material/bar_chart:", expanded=True):
        st.markdown("**Data Parameter Klinis Pasien:**")
        st.dataframe(pd.DataFrame([hasil['input_raw']]), use_container_width=True, hide_index=True)
        
        st.write("")
        st.markdown("**Bobot Tingkat Kepentingan Fitur (Gini Feature Importance):**")
        
        df_fi_sorted = hasil['feature_importance'].sort_values(by='Tingkat Kepentingan', ascending=False).copy()
        df_fi_display = pd.DataFrame({
            'Variabel Klinis': df_fi_sorted['Variabel'],
            'Bobot (Skor)': df_fi_sorted['Tingkat Kepentingan'].map(lambda x: f"{x:.4f}"),
            'Kontribusi (%)': df_fi_sorted['Tingkat Kepentingan'].map(lambda x: f"{x*100:.2f}%")
        })

        col_chart, col_table = st.columns([1.45, 1])
        with col_chart:
            st.bar_chart(
                data=df_fi_sorted,
                x='Variabel',
                y='Tingkat Kepentingan',
                color="#4F46E5",
                use_container_width=True
            )
        with col_table:
            st.dataframe(df_fi_display, use_container_width=True, hide_index=True)

        st.caption(
            "Keterangan Model: Total seluruh bobot fitur bernilai 1,0000 (100%). Variabel dengan nilai 0,0000 (0,00%) tidak terpilih "
            "sebagai titik percabangan pohon keputusan saat pelatihan model karena 5 variabel utama di atas telah memisahkan kelas risiko secara optimal."
        )

    st.write("")
    pdf_bytes = generate_pdf_bytes(hasil)
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("Input Pasien Baru", icon=":material/person_add:", type="primary", use_container_width=True):
            st.session_state['hasil_prediksi'] = None
            st.session_state['nav_menu'] = "Form Input Data"
            st.rerun()
    with col_b2:
        st.download_button(
            label="Unduh Laporan (.pdf)",
            icon=":material/picture_as_pdf:",
            data=pdf_bytes,
            file_name=f"Laporan_Skrining_{hasil['no_rm']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col_b3:
        if st.button("Kembali ke Dashboard", icon=":material/dashboard:", type="secondary", use_container_width=True):
            st.session_state['nav_menu'] = "Dashboard Utama"
            st.rerun()

# ==============================================================================
# 5. HALAMAN RIWAYAT SKRINING (BERSIH TANPA KPI CARDS & TOMBOL TAMBAH PASIEN)
# ==============================================================================
def show_riwayat_page():
    riwayat = ambil_semua_riwayat_db()
    total_pasien = len(riwayat)

    riwayat_banner_html = (
        f'<div class="subpage-banner">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">'
        f'<div style="display: flex; align-items: center; gap: 16px;">'
        f'<div style="background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35); width: 52px; height: 52px; border-radius: 15px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">{ICON_HISTORY}</div>'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 800; letter-spacing: 0.6px; opacity: 0.9;">CLOUD DATABASE • TERSIMPAN PERMANEN</div>'
        f'<h1 style="margin: 2px 0 4px 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.4px;">Riwayat Skrining Pasien</h1>'
        f'<p style="margin: 0; font-size: 0.88rem; opacity: 0.92;">Rekapitulasi pasien yang tersimpan permanen di server Cloud, pencarian data cepat, dan ekspor laporan Excel.</p>'
        f'</div></div>'
        f'<div class="step-pill">Total Tercatat: {total_pasien} Pasien</div>'
        f'</div></div>'
    )
    st.markdown(riwayat_banner_html, unsafe_allow_html=True)

    if not riwayat:
        with st.container(border=True):
            st.markdown(
                '<div style="text-align: center; padding: 28px 16px;">'
                '<div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin-bottom: 6px;">Belum Ada Data Riwayat Skrining di Cloud</div>'
                '<div style="font-size: 0.9rem; color: #64748B;">Silakan lakukan pemeriksaan pasien melalui menu Form Input Data pada navigasi samping untuk menyimpan data ke database.</div>'
                '</div>',
                unsafe_allow_html=True
            )
    else:
        df_riwayat = pd.DataFrame(riwayat)
        with st.container(border=True):
            st.markdown(
                '<div class="section-strip">'
                '<span class="section-badge-indigo">REKAPITULASI CLOUD</span>'
                '<span style="font-size: 1.08rem; font-weight: 800; color: #0F172A;">Daftar Hasil Pemeriksaan Pasien</span>'
                '</div>',
                unsafe_allow_html=True
            )

            col_search, col_filter = st.columns([1.6, 1])
            with col_search:
                search_query = st.text_input("Cari Pasien", placeholder="Ketik Nama Pasien atau No. RM...", label_visibility="collapsed")
            with col_filter:
                filter_risiko = st.selectbox("Filter Kategori", ["Semua Kategori Risiko", "Risiko Tinggi", "Risiko Sedang", "Risiko Rendah"], label_visibility="collapsed")

            df_filtered = df_riwayat.copy()
            if search_query.strip():
                q = search_query.strip().lower()
                df_filtered = df_filtered[
                    df_filtered['Nama Pasien'].str.lower().str.contains(q, na=False) |
                    df_filtered['No. RM'].str.lower().str.contains(q, na=False)
                ]
            if filter_risiko != "Semua Kategori Risiko":
                df_filtered = df_filtered[df_filtered['Hasil Risiko'] == filter_risiko]

            st.dataframe(df_filtered.drop(columns=['ID']), use_container_width=True, hide_index=True)

            st.write("")
            excel_bytes = generate_excel_bytes(df_riwayat.drop(columns=['ID']))
            st.download_button(
                label="Ekspor Rekap ke Excel (.xlsx)",
                icon=":material/table_view:",
                data=excel_bytes,
                file_name=f"Rekap_CardioScreen_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with st.expander("Kelola / Hapus Data Riwayat (Jika Ada Kesalahan Input)", icon=":material/delete_sweep:", expanded=False):
            opsi_hapus = {f"ID #{r['ID']} — {r['No. RM']} ({r['Nama Pasien']}) [{r['Waktu']}]": r['ID'] for r in riwayat}
            pilihan_label = st.selectbox("Pilih data pasien yang ingin dihapus dari Cloud Database:", options=list(opsi_hapus.keys()))
            if st.button("Hapus Data Terpilih", icon=":material/delete:", type="secondary"):
                hapus_riwayat_by_id(opsi_hapus[pilihan_label])
                st.success("Data berhasil dihapus permanen dari Cloud Database.")
                st.rerun()

# ==============================================================================
# MAIN ROUTING & MODERN SIDEBAR NAVIGATION
# ==============================================================================
def main():
    if not st.session_state['logged_in']:
        show_login_page()
    else:
        sidebar_brand_html = (
            f'<div style="display: flex; align-items: center; gap: 12px; padding: 4px 4px 8px 4px;">'
            f'{BRAND_LOGO_SVG}'
            f'<div><div style="color:#0F172A; font-weight:800; font-size:1.1rem; line-height:1.1; letter-spacing:-0.3px;">CARDIOSCREEN</div>'
            f'<div style="color:#4F46E5; font-size:0.76rem; font-weight:700; margin-top:2px;">Puskesmas Bandung</div></div>'
            f'</div>'
        )
        st.sidebar.markdown(sidebar_brand_html, unsafe_allow_html=True)
        
        st.sidebar.divider()
        st.sidebar.caption("NAVIGASI UTAMA")

        nav_items = [
            ("Dashboard Utama", ":material/dashboard:"),
            ("Form Input Data", ":material/clinical_notes:"),
            ("Hasil Klasifikasi", ":material/analytics:"),
            ("Riwayat Skrining", ":material/history:")
        ]

        for label, icon_name in nav_items:
            is_active = (st.session_state['nav_menu'] == label)
            if st.sidebar.button(
                label,
                key=f"nav_{label}",
                icon=icon_name,
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                st.session_state['nav_menu'] = label
                st.rerun()
        
        st.sidebar.divider()
        petugas_box_html = (
            f'<div style="background: linear-gradient(135deg, #EEF2FF 0%, #FFF1F2 100%); border: 1.5px solid #C7D2FE; border-radius: 15px; padding: 12px 14px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);">'
            f'<div style="font-size: 0.68rem; color: #4338CA; font-weight: 800; letter-spacing: 0.5px;">PETUGAS AKTIF</div>'
            f'<div style="font-size: 0.95rem; color: #0F172A; font-weight: 800; margin-top: 2px;">{st.session_state["username"]}</div>'
            f'</div>'
        )
        st.sidebar.markdown(petugas_box_html, unsafe_allow_html=True)

        if st.sidebar.button("Keluar Sistem", icon=":material/logout:", type="secondary", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['hasil_prediksi'] = None
            st.session_state['nav_menu'] = "Dashboard Utama"
            st.rerun()

        if st.session_state['nav_menu'] == "Dashboard Utama":
            show_dashboard_page()
        elif st.session_state['nav_menu'] == "Form Input Data":
            show_input_page()
        elif st.session_state['nav_menu'] == "Hasil Klasifikasi":
            show_hasil_page()
        elif st.session_state['nav_menu'] == "Riwayat Skrining":
            show_riwayat_page()

if __name__ == "__main__":
    main()