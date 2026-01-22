import streamlit as st
import pandas as pd
import joblib

# ==========================================
# CONFIGURATION PAGE
# ==========================================
st.set_page_config(
    page_title="Prédiction du Risque Cardiaque (CHD)",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# STYLE UNIQUE (SOBRE & MODERNE)
# ==========================================
st.markdown("""
<style>
/* Fond général */
.main {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

/* Container */
.block-container {
    padding-top: 2.5rem;
}

/* Cartes */
.card {
    background: white;
    padding: 28px;
    border-radius: 14px;
    border-left: 6px solid #ef4444;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 24px;
}

/* Carte résultat */
.result-card {
    border-left: 6px solid #2563eb;
}

/* Titres */
h1 {
    font-weight: 700;
    color: #0f172a;
}
.subtitle {
    color: #475569;
    margin-bottom: 30px;
}

/* Texte centré */
.center {
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Bouton */
div.stButton > button {
    background-color: #ef4444;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
}
div.stButton > button:hover {
    background-color: #dc2626;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CHARGEMENT DU MODÈLE
# ==========================================
model = joblib.load("Model.pkl")

# ==========================================
# SIDEBAR – PARAMÈTRES CLINIQUES
# ==========================================
st.sidebar.title("🩺 Paramètres Cliniques")

sbp = st.sidebar.number_input("Pression sanguine (sbp)", 80, 250, 130)
tobacco = st.sidebar.number_input("Tabac (kg cumulé)", 0.0, value=1.0)
ldl = st.sidebar.number_input("LDL Cholestérol", 0.0, value=4.0)
adiposity = st.sidebar.number_input("Adiposité", 0.0, value=25.0)

famhist = st.sidebar.selectbox(
    "Antécédents familiaux (famhist)",
    ["Absent", "Present"]
)

typea = st.sidebar.number_input("Comportement Type A", 0.0, value=50.0)
obesity = st.sidebar.number_input("Obésité", 0.0, value=25.0)
alcohol = st.sidebar.number_input("Consommation Alcool", 0.0, value=10.0)
age = st.sidebar.number_input("Âge", 20, 100, 45)

# ==========================================
# CONTENU PRINCIPAL
# ==========================================
st.markdown("<h1 class='center'> Prédiction du Risque Cardiaque (CHD)</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='center subtitle'>Application basée sur un modèle de Machine Learning pour estimer le risque de maladie cardiaque.</p>",
    unsafe_allow_html=True
)

# ==========================================
# CARTE DONNÉES PATIENT
# ==========================================
input_data = pd.DataFrame([{
    "sbp": sbp,
    "tobacco": tobacco,
    "ldl": ldl,
    "adiposity": adiposity,
    "famhist": famhist,
    "typea": typea,
    "obesity": obesity,
    "alcohol": alcohol,
    "age": age
}])

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Données patient saisies")
st.dataframe(input_data, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# BOUTON DE PRÉDICTION
# ==========================================
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict = st.button("🧠 Lancer le diagnostic")

# ==========================================
# RÉSULTAT
# ==========================================
if predict:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("<div class='card result-card center'>", unsafe_allow_html=True)
    st.subheader("📊 Résultat de la prédiction")

    if prediction == 1:
        st.error("⚠️ Risque cardiaque ÉLEVÉ")
    else:
        st.success("✅ Risque cardiaque FAIBLE")

    st.markdown(
        f"<h2>{probability*100:.2f} %</h2>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.caption("Projet académique – Prédiction du Risque Cardiaque (Machine Learning)")
