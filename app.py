from pathlib import Path
import shutil, zipfile, textwrap, ast

out = Path("/mnt/data/pcos_render_app")
out.mkdir(exist_ok=True)

app_code = r'''import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Hide harmless model-version warnings in the UI.
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "PCOS_XGBoost_Final_Model.joblib"


# ============================================================
# Load the saved model package
# ============================================================
@st.cache_resource
def load_package():
    package = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "feature_columns",
        "threshold",
    }

    missing = required_keys - set(package.keys())
    if missing:
        raise ValueError(
            f"Model package is missing required keys: {sorted(missing)}"
        )

    return package


# ============================================================
# Exact feature names from the training notebook
# ============================================================
FEATURES = [
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "BMI",
    "Blood Group",
    "Pulse rate(bpm)",
    "RR (breaths/min)",
    "Hb(g/dl)",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Marraige Status (Yrs)",
    "Pregnant(Y/N)",
    "No. of aborptions",
    "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "FSH/LH",
    "Hip(inch)",
    "Waist(inch)",
    "Waist:Hip Ratio",
    "TSH (mIU/L)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food(Y/N)",
    "Reg.Exercise(Y/N)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
]

# These are the categorical columns used by the final training pipeline.
CATEGORICAL_COLUMNS = [
    "Blood Group",
    "Cycle(R/I)",
    "Pregnant(Y/N)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food(Y/N)",
    "Reg.Exercise(Y/N)",
]

# Derived exactly from the cleaned training data.
DERIVED_COLUMNS = {
    "BMI",
    "FSH/LH",
    "Waist:Hip Ratio",
}

# Dataset categorical codes observed in the supplied training data.
BLOOD_GROUP_CODES = [11, 12, 13, 14, 15, 16, 17, 18]
CYCLE_CODES = [2, 4, 5]


# ============================================================
# Page
# ============================================================
st.set_page_config(
    page_title="PCOS Prediction",
    page_icon="🩺",
    layout="centered",
)

st.title("PCOS Prediction")
st.caption("XGBoost-based prediction using your trained model.")

try:
    package = load_package()
    model = package["model"]
    model_features = list(package["feature_columns"])
    threshold = float(package["threshold"])
except Exception as e:
    st.error("The model could not be loaded.")
    st.exception(e)
    st.stop()

# Safety check: make sure the UI matches the saved model.
if model_features != FEATURES:
    st.error(
        "Feature mismatch detected between the saved model and this UI. "
        "The application was stopped to prevent an incorrect prediction."
    )
    st.write("Model features:", model_features)
    st.write("UI features:", FEATURES)
    st.stop()


# ============================================================
# Patient / general information
# ============================================================
with st.expander("1. Basic information", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age (years)",
            min_value=10.0,
            max_value=80.0,
            value=31.0,
            step=1.0,
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=59.0,
            step=0.1,
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=220.0,
            value=156.0,
            step=0.1,
        )

        pulse = st.number_input(
            "Pulse rate (bpm)",
            min_value=30.0,
            max_value=200.0,
            value=72.0,
            step=1.0,
        )

        rr = st.number_input(
            "RR (breaths/min)",
            min_value=5.0,
            max_value=60.0,
            value=18.0,
            step=1.0,
        )

        hb = st.number_input(
            "Hb (g/dl)",
            min_value=1.0,
            max_value=25.0,
            value=11.0,
            step=0.1,
        )

    with c2:
        blood_group = st.selectbox(
            "Blood Group code",
            BLOOD_GROUP_CODES,
            help="The training dataset stores Blood Group as numeric category codes 11–18. No external blood-group mapping was provided.",
        )

        cycle = st.selectbox(
            "Cycle code",
            CYCLE_CODES,
            help="The supplied training data contains the numeric codes 2, 4 and 5 for Cycle(R/I).",
        )

        cycle_length = st.number_input(
            "Cycle length (days)",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )

        marriage_years = st.number_input(
            "Marriage status (years)",
            min_value=0.0,
            max_value=80.0,
            value=7.0,
            step=1.0,
        )

        pregnant = st.selectbox(
            "Pregnant (Y/N)",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

        abortions = st.number_input(
            "Number of abortions",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
        )


# ============================================================
# Hormonal / laboratory information
# ============================================================
with st.expander("2. Hormonal & laboratory values"):
    c1, c2 = st.columns(2)

    with c1:
        beta_hcg_1 = st.number_input(
            "I beta-HCG (mIU/mL)",
            min_value=0.0,
            value=20.0,
            step=0.01,
        )

        beta_hcg_2 = st.number_input(
            "II beta-HCG (mIU/mL)",
            min_value=0.0,
            value=1.99,
            step=0.01,
        )

        fsh = st.number_input(
            "FSH (mIU/mL)",
            min_value=0.0,
            value=4.85,
            step=0.01,
        )

        lh = st.number_input(
            "LH (mIU/mL)",
            min_value=0.01,
            value=2.30,
            step=0.01,
        )

        tsh = st.number_input(
            "TSH (mIU/L)",
            min_value=0.0,
            value=2.26,
            step=0.01,
        )

        amh = st.number_input(
            "AMH (ng/mL)",
            min_value=0.0,
            value=3.70,
            step=0.01,
        )

    with c2:
        prl = st.number_input(
            "PRL (ng/mL)",
            min_value=0.0,
            value=21.92,
            step=0.01,
        )

        vit_d3 = st.number_input(
            "Vit D3 (ng/mL)",
            min_value=0.0,
            value=25.90,
            step=0.01,
        )

        prg = st.number_input(
            "PRG (ng/mL)",
            min_value=0.0,
            value=0.32,
            step=0.01,
        )

        rbs = st.number_input(
            "RBS (mg/dl)",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

        systolic = st.number_input(
            "BP Systolic (mmHg)",
            min_value=50.0,
            max_value=250.0,
            value=110.0,
            step=1.0,
        )

        diastolic = st.number_input(
            "BP Diastolic (mmHg)",
            min_value=30.0,
            max_value=180.0,
            value=80.0,
            step=1.0,
        )


# ============================================================
# Body measurements
# ============================================================
with st.expander("3. Body measurements"):
    c1, c2 = st.columns(2)

    with c1:
        hip = st.number_input(
            "Hip (inch)",
            min_value=20.0,
            max_value=80.0,
            value=38.0,
            step=0.1,
        )

        waist = st.number_input(
            "Waist (inch)",
            min_value=20.0,
            max_value=80.0,
            value=34.0,
            step=0.1,
        )

        bmi = (
            weight / ((height / 100.0) ** 2)
            if height > 0
            else 0.0
        )

        waist_hip_ratio = waist / hip if hip > 0 else 0.0

        st.metric("BMI (calculated)", f"{bmi:.2f}")
        st.metric("Waist : Hip Ratio (calculated)", f"{waist_hip_ratio:.3f}")

    with c2:
        fsh_lh = fsh / lh if lh > 0 else 0.0

        st.metric("FSH / LH (calculated)", f"{fsh_lh:.3f}")

        st.caption(
            "BMI, FSH/LH and Waist:Hip Ratio are calculated automatically "
            "from the corresponding input values, matching the training notebook."
        )


# ============================================================
# Symptoms / lifestyle
# ============================================================
with st.expander("4. Symptoms & lifestyle"):
    c1, c2 = st.columns(2)

    yes_no = [0, 1]

    with c1:
        weight_gain = st.selectbox(
            "Weight gain (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

        hair_growth = st.selectbox(
            "Hair growth (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

        skin_darkening = st.selectbox(
            "Skin darkening (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

        hair_loss = st.selectbox(
            "Hair loss (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

    with c2:
        pimples = st.selectbox(
            "Pimples (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

        fast_food = st.selectbox(
            "Fast food (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )

        exercise = st.selectbox(
            "Regular exercise (Y/N)",
            yes_no,
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
        )


# ============================================================
# Ultrasound / follicle information
# ============================================================
with st.expander("5. Ultrasound / follicle information"):
    c1, c2 = st.columns(2)

    with c1:
        follicle_left = st.number_input(
            "Follicle No. (L)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )

        avg_f_size_left = st.number_input(
            "Avg. F size (L) (mm)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.1,
        )

    with c2:
        follicle_right = st.number_input(
            "Follicle No. (R)",
            min_value=0.0,
            max_value=100.0,
            value=6.0,
            step=1.0,
        )

        avg_f_size_right = st.number_input(
            "Avg. F size (R) (mm)",
            min_value=0.0,
            max_value=100.0,
            value=16.0,
            step=0.1,
        )

    endometrium = st.number_input(
        "Endometrium (mm)",
        min_value=0.0,
        max_value=50.0,
        value=8.5,
        step=0.1,
    )


# ============================================================
# Prediction
# ============================================================
st.divider()

predict = st.button(
    "Predict PCOS",
    type="primary",
    use_container_width=True,
)

if predict:
    patient = {
        "Age (yrs)": float(age),
        "Weight (Kg)": float(weight),
        "Height(Cm)": float(height),
        "BMI": float(bmi),
        "Blood Group": int(blood_group),
        "Pulse rate(bpm)": float(pulse),
        "RR (breaths/min)": float(rr),
        "Hb(g/dl)": float(hb),
        "Cycle(R/I)": int(cycle),
        "Cycle length(days)": float(cycle_length),
        "Marraige Status (Yrs)": float(marriage_years),
        "Pregnant(Y/N)": int(pregnant),
        "No. of aborptions": float(abortions),
        "I   beta-HCG(mIU/mL)": float(beta_hcg_1),
        "II    beta-HCG(mIU/mL)": float(beta_hcg_2),
        "FSH(mIU/mL)": float(fsh),
        "LH(mIU/mL)": float(lh),
        "FSH/LH": float(fsh_lh),
        "Hip(inch)": float(hip),
        "Waist(inch)": float(waist),
        "Waist:Hip Ratio": float(waist_hip_ratio),
        "TSH (mIU/L)": float(tsh),
        "AMH(ng/mL)": float(amh),
        "PRL(ng/mL)": float(prl),
        "Vit D3 (ng/mL)": float(vit_d3),
        "PRG(ng/mL)": float(prg),
        "RBS(mg/dl)": float(rbs),
        "Weight gain(Y/N)": int(weight_gain),
        "hair growth(Y/N)": int(hair_growth),
        "Skin darkening (Y/N)": int(skin_darkening),
        "Hair loss(Y/N)": int(hair_loss),
        "Pimples(Y/N)": int(pimples),
        "Fast food(Y/N)": int(fast_food),
        "Reg.Exercise(Y/N)": int(exercise),
        "BP _Systolic (mmHg)": float(systolic),
        "BP _Diastolic (mmHg)": float(diastolic),
        "Follicle No. (L)": float(follicle_left),
        "Follicle No. (R)": float(follicle_right),
        "Avg. F size (L) (mm)": float(avg_f_size_left),
        "Avg. F size (R) (mm)": float(avg_f_size_right),
        "Endometrium (mm)": float(endometrium),
    }

    patient_df = pd.DataFrame([patient], columns=model_features)

    try:
        probability = float(model.predict_proba(patient_df)[0, 1])
        prediction = int(probability >= threshold)

        st.subheader("Result")

        if prediction == 1:
            st.error("PCOS: YES")
        else:
            st.success("PCOS: NO")

        st.metric(
            "PCOS probability",
            f"{probability * 100:.2f}%",
        )

        st.caption(
            f"Model threshold: {threshold:.2f}"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.caption(
    "For research/demo use only. This is not a medical diagnosis."
)
'''

requirements = """streamlit>=1.40,<2
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn==1.6.1
xgboost==3.1.3
joblib>=1.3,<2
"""

readme = """# PCOS Prediction - Render

Minimal Streamlit UI for the supplied XGBoost PCOS model.

## Files

- `app.py`
- `PCOS_XGBoost_Final_Model.joblib`
- `requirements.txt`

## Render settings

Build Command:
```bash
pip install -r requirements.txt
