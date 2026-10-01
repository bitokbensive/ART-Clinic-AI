import streamlit as st
from pathlib import Path
import joblib
import json

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI-Powered ART Clinic",
    page_icon="🩺",
    layout="wide"
)

# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "ART_interruption_logistic_model_final.joblib"
METADATA_PATH = MODEL_DIR / "ART_interruption_model_metadata_v1.json"

# ============================================
# HEADER
# ============================================

st.title("🩺 AI-Powered ART Clinic")
st.subheader("ART Care Interruption Risk Assessment")

st.write(
    "Machine-learning decision-support prototype for identifying "
    "patients who may require additional ART care follow-up."
)

# ============================================
# MODEL LOADING
# ============================================

st.header("System Status")

# Check model folder
if not MODEL_DIR.exists():

    st.error("❌ Model folder not found.")

    st.write("Expected location:")
    st.code(str(MODEL_DIR))

    st.stop()

st.success("✅ Model folder found.")

# Check model file
if not MODEL_PATH.exists():

    st.error("❌ Model file not found.")

    st.write("Expected model:")
    st.code(str(MODEL_PATH))

    st.write("Files currently inside the model folder:")

    for file in MODEL_DIR.iterdir():
        st.write(file.name)

    st.stop()

st.success("✅ Model file found.")

# Load model
try:

    model = joblib.load(MODEL_PATH)

    st.success("✅ Model loaded successfully.")

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.exception(e)

    st.stop()

# ============================================
# METADATA
# ============================================

metadata = None

if METADATA_PATH.exists():

    try:

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        st.success("✅ Model metadata loaded.")

    except Exception as e:

        st.warning("⚠️ Metadata file could not be loaded.")
        st.exception(e)

else:

    st.warning(
        "⚠️ Model metadata file was not found. "
        "The model itself is still available."
    )

# ============================================
# MODEL INFORMATION
# ============================================

st.header("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Training Cohort",
        "1,341"
    )

with col3:
    st.metric(
        "OOF ROC-AUC",
        "0.821"
    )

st.info(
    "This is a research/prototype decision-support system. "
    "It does not replace clinical judgement."
)

# ============================================
# PATIENT ASSESSMENT
# ============================================

st.header("Patient Risk Assessment")

st.write(
    "Enter the patient's model variables below to calculate "
    "an ART care interruption risk score."
)

# ============================================
# INPUTS
# ============================================

col1, col2, col3 = st.columns(3)

with col1:

    age_at_art = st.number_input(
        "Age at ART initiation",
        min_value=0,
        max_value=120,
        value=30,
        step=1
    )

    bCD4Count = st.number_input(
        "Baseline CD4 count",
        min_value=0.0,
        max_value=3000.0,
        value=500.0,
        step=1.0
    )

    art_start_year = st.number_input(
        "ART start year",
        min_value=1980,
        max_value=2030,
        value=2022,
        step=1
    )

with col2:

    Gender = st.selectbox(
        "Gender",
        [
            "F",
            "M"
        ]
    )

    population_type = st.selectbox(
        "Population type",
        [
            "Key Population",
            "General Population"
        ]
    )

    key_population_type = st.selectbox(
        "Key population type",
        [
            "Female sex worker",
            "Unknown",
            "Men who have sex with men"
        ]
    )

with col3:

    EnrollmentType = st.selectbox(
        "Enrollment type",
        [
            "New client",
            "Transfer In",
            "Re-enroll",
            "Missing"
        ]
    )

    StartARTRegimenLine = st.selectbox(
        "ART regimen line",
        [
            "First line",
            "Second line"
        ]
    )

    bWHOStage = st.selectbox(
        "WHO stage",
        [
            "WHO STAGE 1 ADULT",
            "WHO STAGE 2 ADULT",
            "WHO STAGE 3 ADULT",
            "WHO STAGE 4 ADULT"
        ]
    )

StartARTRegimen = st.selectbox(
    "Starting ART regimen",
    [
        "AZT+3TC+NVP",
        "3TC+NVP+TDF",
        "TDF+3TC+EFV",
        "TDF+3TC+DTG",
        "AZT+3TC+EFV",
        "AZT+3TC+ATV/r",
        "TDF+3TC+ATV/r",
        "D4T+3TC+NVP",
        "TDF+3TC+LPV/r",
        "D4T+3TC+EFV",
        "ABC+3TC+DTG",
        "ABC+3TC+EFV",
        "ABC+3TC+NVP"
    ]
)

# ============================================
# PREDICTION
# ============================================

st.divider()

if st.button(
    "🔍 Calculate ART Interruption Risk",
    type="primary",
    use_container_width=True
):

    import pandas as pd

    patient_data = pd.DataFrame([
        {
            "age_at_art": age_at_art,
            "bCD4Count": bCD4Count,
            "art_start_year": art_start_year,
            "Gender": Gender,
            "population_type": population_type,
            "key_population_type": key_population_type,
            "EnrollmentType": EnrollmentType,
            "StartARTRegimen": StartARTRegimen,
            "StartARTRegimenLine": StartARTRegimenLine,
            "bWHOStage": bWHOStage
        }
    ])

    try:

        risk = model.predict_proba(patient_data)[0, 1]

        st.header("Risk Assessment")

        st.metric(
            "Predicted ART Care Interruption Risk",
            f"{risk:.1%}"
        )

        # Risk category
        if risk >= 0.70:

            st.error(
                "🔴 HIGH PRIORITY"
            )

            st.write(
                "Consider prioritizing this patient for appropriate "
                "follow-up and clinical review."
            )

        elif risk >= 0.40:

            st.warning(
                "🟠 MODERATE PRIORITY"
            )

            st.write(
                "Consider additional review and follow-up according "
                "to clinic procedures."
            )

        else:

            st.success(
                "🟢 LOW PRIORITY"
            )

            st.write(
                "Continue routine ART care and monitoring."
            )

        st.caption(
            "The score represents model-estimated risk, not a diagnosis "
            "or confirmation that ART interruption will occur."
        )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)

# ============================================
# FOOTER
# ============================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        line-height: 1.6;
        padding: 20px 0 10px 0;
    ">
        <strong>AI-Powered ART Clinic</strong><br>
        Developed by <strong>Bensive J Bitok</strong><br>
        <strong>Organization:</strong> PHDA Kariobangi SWOP Site<br>
        <strong>Programme:</strong> ENGAGE<br>
        <strong>Model Version:</strong> v1.0.0<br>
        © 2026
        <br><br>
        <em>Research / Decision-Support Prototype</em><br>
        <em>
        For clinical decision support only.
        Not a substitute for professional clinical judgement.
        </em>
    </div>
    """,
    unsafe_allow_html=True
)