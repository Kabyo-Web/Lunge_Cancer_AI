from download_models import download_models
download_models()
import streamlit as st
from PIL import Image
import tempfile
import predict
from fpdf import FPDF
import datetime
import time

st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .result-normal {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white;
        font-size: 24px; font-weight: bold;
        margin: 10px 0;
    }
    .result-benign {
        background: linear-gradient(135deg, #7b4f00, #c97d00);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white;
        font-size: 24px; font-weight: bold;
        margin: 10px 0;
    }
    .result-malignant {
        background: linear-gradient(135deg, #7b0000, #c0392b);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white;
        font-size: 24px; font-weight: bold;
        margin: 10px 0;
    }
    .result-uncertain {
        background: linear-gradient(135deg, #1a1a4e, #2e2e8e);
        padding: 20px; border-radius: 12px;
        text-align: center; color: white;
        font-size: 24px; font-weight: bold;
        margin: 10px 0;
    }
    .confidence-box {
        background: #1e2130;
        padding: 15px; border-radius: 10px;
        text-align: center; color: #a0aec0;
        font-size: 18px; margin: 10px 0;
    }
    .history-card {
        background: #1e2130;
        padding: 12px; border-radius: 10px;
        margin: 8px 0; color: white;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### 📋 Prediction History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            label = h['label']
            if label == "Normal":
                icon = "🟢"
            elif label == "Benign":
                icon = "🟡"
            elif label == "Malignant":
                icon = "🔴"
            else:
                icon = "🔵"  # Uncertain
            st.markdown(f"""
            <div class="history-card">
                {icon} <b>{h['file']}</b><br>
                🔬 {label} | 📊 {h['confidence']}<br>
                🕐 {h['time']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No predictions yet.")

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("# 🫁 Lung Cancer Detection AI")
st.markdown("Upload a lung CT scan image to get an AI-powered prediction.")
st.markdown("---")

tab1, tab2 = st.tabs(["🔬 Analyze", "ℹ️ About"])

with tab1:
    uploaded = st.file_uploader("Upload CT Scan", type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)  # ✅ fixed deprecation

        with col2:
            analyze = st.button("🔍 Analyze", use_container_width=True)

            if analyze:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp.name)
                tmp.close()

                with st.spinner("🔍 Analyzing CT scan..."):
                    time.sleep(1)
                    try:
                        label, confidence = predict.predict_image(tmp.name)

                        if label == "Normal":
                            st.markdown('<div class="result-normal">✅ Normal</div>', unsafe_allow_html=True)
                        elif label == "Benign":
                            st.markdown('<div class="result-benign">⚠️ Benign</div>', unsafe_allow_html=True)
                        elif label == "Malignant":
                            st.markdown('<div class="result-malignant">🚨 Malignant</div>', unsafe_allow_html=True)
                        else:  # Uncertain
                            st.markdown('<div class="result-uncertain">🔵 Uncertain</div>', unsafe_allow_html=True)
                            st.info("⚠️ Model confidence is too low (< 65%). This image may be ambiguous. Please consult a medical professional.")

                        st.markdown(f'<div class="confidence-box">📊 Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
                        st.progress(confidence)

                        st.session_state.history.append({
                            "file": uploaded.name,
                            "label": label,
                            "confidence": f"{confidence*100:.2f}%",
                            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # PDF Report
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", "B", size=18)
                        pdf.cell(200, 12, txt="Lung Cancer AI Report", ln=True, align="C")
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                        pdf.cell(200, 10, txt=f"File: {uploaded.name}", ln=True)
                        pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
                        pdf.cell(200, 10, txt=f"Confidence: {confidence*100:.2f}%", ln=True)
                        if label == "Uncertain":
                            pdf.cell(200, 10, txt="Note: Low confidence prediction. Further evaluation recommended.", ln=True)
                        pdf.ln(10)
                        pdf.set_font("Arial", "I", size=10)
                        pdf.multi_cell(200, 8, txt="Disclaimer: This AI prediction is for research purposes only. Please consult a medical professional for diagnosis.")

                        pdf_path = tempfile.mktemp(suffix=".pdf")
                        pdf.output(pdf_path)

                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📄 Download Report (PDF)",
                                data=f,
                                file_name="lung_cancer_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                    except Exception as e:
                        if "NOT_CT_SCAN" in str(e):
                            st.error("❌ This is not a valid CT Scan image. Please upload a proper lung CT scan.")
                        else:
                            st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.markdown("""
    ## About This App

    This AI-powered application detects lung cancer from CT scan images using a hybrid deep learning model.

    ### How it works
    - Upload a lung CT scan image
    - Click **Analyze** button
    - The AI analyzes the image using EfficientNet + SVM

    ### Classes
    | Class | Description |
    |-------|-------------|
    | 🟢 Normal | No signs of cancer detected |
    | 🟡 Benign | Non-cancerous tumor detected |
    | 🔴 Malignant | Cancerous tumor detected |
    | 🔵 Uncertain | Confidence too low, further evaluation needed |

    ### Disclaimer
    > This tool is for **research and educational purposes only**.
    > Always consult a qualified medical professional for diagnosis.

    ### Model
    - Feature Extractor: EfficientNet B0
    - Classifier: Support Vector Machine (SVM)
    - Image Size: 128x128
    """)