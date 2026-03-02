import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from sqlalchemy import create_engine, text
from streamlit_option_menu import option_menu
import tempfile
import os

# --- 1. PAGE CONFIGURATION & CSS (LIGHT THEME) ---
st.set_page_config(
    page_title="NeuroRetina AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Fixes for Input Boxes & Visuals
st.markdown("""
<style>
    /* Root app background */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e6e6e6;
    }
    /* Headings & Text */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #000000 !important;
        font-family: 'Helvetica Neue', Arial, sans-serif !important;
    }
    .stMarkdown, .stText {
        color: #000000 !important;
    }
    /* Metric Cards (light) */
    div[data-testid="metric-container"] {
        background-color: #fafafa !important;
        border: 1px solid #e6e6e6 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
        color: #000000 !important;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #1976d2 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: 0.2s !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #125aa3 !important;
        box-shadow: 0px 6px 18px rgba(25, 118, 210, 0.18) !important;
    }
    /* Input Styling */
    .stTextInput > div > div {
        background-color: #ffffff !important;
        border: 1px solid #dcdcdc !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    input[type="text"], input[type="password"] {
        border: none !important;
        background-color: transparent !important;
        color: #000000 !important;
        padding: 0px !important;
    }
    textarea, select, .stDateInput > div > div {
        background-color: #ffffff !important;
        border: 1px solid #dcdcdc !important;
        color: #000000 !important;
        border-radius: 8px !important;
    }
    /* File uploader box */
    .css-1p3svff, .css-1fcb6tp { 
        background-color: #ffffff !important;
        border: 1px dashed #dcdcdc !important;
        color: #000000 !important;
    }
    /* Option menu selected style */
    .nav-link-selected {
        background-color: #1976d2 !important;
        color: #ffffff !important;
    }
    .nav-link {
        color: #000000 !important;
    }
    /* Chat / messages */
    [data-testid="stChatMessage"] {
        color: #000000 !important;
    }
    /* Plotly background */
    .plotly-graph-div {
        background: transparent !important;
    }
    /* Dataframe */
    .stDataFrame, .stDataEditor {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* PDF download button style fixes */
    button[title="Download file"] {
        background-color: #1976d2 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
IMAGE_DIR = Path("app_data/images")
DB_FILE = Path("app_data/records.db")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_ENGINE = create_engine(f"sqlite:///{DB_FILE}")
CLASS_NAMES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']

# --- MEDICAL KNOWLEDGE BASE ---
MEDICAL_DB = {
    "AMD": {"name": "Age-Related Macular Degeneration", "desc": "Macula disorder causing blurred central vision.", "treatment": "Anti-VEGF injections, Photodynamic therapy.", "action": "Refer to Retinal Specialist."},
    "CNV": {"name": "Choroidal Neovascularization", "desc": "Growth of abnormal blood vessels beneath the retina.", "treatment": "Urgent Anti-VEGF therapy.", "action": "URGENT referral required."},
    "CSR": {"name": "Central Serous Retinopathy", "desc": "Fluid accumulation under the retina.", "treatment": "Observation, Laser photocoagulation.", "action": "Monitor for 3 months."},
    "DME": {"name": "Diabetic Macular Edema", "desc": "Fluid accumulation in macula due to diabetes.", "treatment": "Anti-VEGF, Focal Laser, Steroids.", "action": "Strict blood sugar control."},
    "DR": {"name": "Diabetic Retinopathy", "desc": "Damage to blood vessels in the retina from diabetes.", "treatment": "Laser surgery, Vitrectomy.", "action": "Diabetes management plan."},
    "DRUSEN": {"name": "Drusen Deposits", "desc": "Yellow deposits under the retina; early sign of AMD.", "treatment": "AREDS2 Vitamins, regular monitoring.", "action": "Routine 6-month checkup."},
    "MH": {"name": "Macular Hole", "desc": "Small break in the macula.", "treatment": "Vitrectomy surgery.", "action": "Surgical referral."},
    "NORMAL": {"name": "Healthy Retina", "desc": "No significant pathological findings.", "treatment": "Routine eye exams.", "action": "None."}
}

# --- BACKEND FUNCTIONS ---
def init_db():
    with DB_ENGINE.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS cases (id INTEGER PRIMARY KEY, patient_id TEXT, date TEXT, prediction TEXT, confidence REAL, notes TEXT)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)"))
        try:
            if conn.execute(text("SELECT * FROM users WHERE username='doctor'")).first() is None:
                conn.execute(text("INSERT INTO users VALUES ('doctor', 'doc123', 'doctor')"))
                conn.commit()
        except: pass

def validate_oct_image(img_bgr):
    """
    Checks if the uploaded image looks like a valid OCT scan.
    Returns: (bool, string_reason)
    """
    # 1. Check Color Saturation (OCT scans are usually Grayscale)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    
    # Threshold: If mean saturation > 30 (out of 255), it has too much color
    if mean_saturation > 30:
        return False, f"Image is too colorful (Saturation: {mean_saturation:.1f}). OCT scans must be grayscale."

    # 2. Check Brightness/Intensity (OCT scans usually have dark backgrounds)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Threshold: If image is very bright (like a white document), reject it.
    if mean_brightness > 180:
        return False, "Image is too bright/white. OCT scans usually have dark backgrounds."
        
    # 3. Variance Check (Solid color images or blank images)
    variance = np.var(gray)
    if variance < 100:
        return False, "Image is too blurry or flat (low variance)."

    return True, "Valid"

@st.cache_resource
def load_clinical_model():
    try:
        custom = {}
        try:
            import tensorflow_addons as tfa
            custom['F1Score'] = tfa.metrics.F1Score
        except: pass
        
        if not Path("Trained_Model.keras").exists():
            return None
        return tf.keras.models.load_model("Trained_Model.keras", custom_objects=custom)
    except:
        return None 

def generate_heatmap(img_array, model):
    try:
        backbone, final_dense = None, None
        for layer in model.layers:
            if "mobilenet" in layer.name.lower(): backbone = layer
            if isinstance(layer, tf.keras.layers.Dense): final_dense = layer
        if not backbone or not final_dense: return None

        last_conv_layer = backbone.get_layer("Conv_1")
        feature_model = tf.keras.models.Model(inputs=backbone.input, outputs=last_conv_layer.output)
        img_pre = tf.keras.applications.mobilenet_v3.preprocess_input(img_array.copy())

        with tf.GradientTape() as tape:
            features = feature_model(img_pre)
            preds = model(img_array)
            top_pred_index = tf.argmax(preds[0])

        W = final_dense.get_weights()[0]
        class_weights = W[:, top_pred_index]
        heatmap = np.dot(features[0], class_weights)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except: return None

# --- PROFESSIONAL PDF GENERATOR ---
class MedicalReport(FPDF):
    def header(self):
        # Brand Title
        self.set_font('Arial', 'B', 20)
        self.set_text_color(25, 118, 210)  # Professional Blue
        self.cell(0, 10, 'NEURO-RETINA DIAGNOSTICS', 0, 1, 'C')
        
        # Subtitle
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 100, 100) # Grey
        self.cell(0, 5, 'Advanced AI-Powered Ophthalmic Analysis System', 0, 1, 'C')
        
        # Divider Line
        self.ln(5)
        self.set_draw_color(25, 118, 210)
        self.set_line_width(0.5)
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-25)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f'Page {self.page_no()}', 0, 1, 'C')
        self.cell(0, 5, 'DISCLAIMER: This report is generated by an Artificial Intelligence system.', 0, 1, 'C')
        self.cell(0, 5, 'It is intended to assist medical professionals.', 0, 1, 'C')

    def section_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(240, 240, 245) # Light Blue-Grey background
        self.cell(0, 8, f"  {label}", 0, 1, 'L', 1)
        self.ln(4)

    def body_text(self, text):
        self.set_font('Arial', '', 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(3)

def generate_pdf(pid, label, conf, notes, img_path, heatmap_path, medical_info):
    pdf = MedicalReport()
    pdf.add_page()
    
    # --- 1. PATIENT DETAILS ---
    pdf.section_title("PATIENT DEMOGRAPHICS")
    pdf.set_font("Arial", "", 11)
    
    # Aligned layout
    pdf.cell(40, 8, "Patient ID:", 0, 0)
    pdf.cell(50, 8, f"{pid}", 0, 0)
    pdf.cell(40, 8, "Scan Date:", 0, 0)
    pdf.cell(0, 8, datetime.now().strftime('%Y-%m-%d %H:%M'), 0, 1)
    
    pdf.cell(40, 8, "Referral:", 0, 0)
    pdf.cell(0, 8, "Dr. User (NeuroRetina)", 0, 1)
    pdf.ln(5)

    # --- 2. DIAGNOSTIC SUMMARY ---
    pdf.section_title("DIAGNOSTIC SUMMARY")
    pdf.set_font("Arial", "B", 14)
    
    if label == "NORMAL":
        pdf.set_text_color(0, 128, 0) # Green
    else:
        pdf.set_text_color(180, 0, 0) # Red
        
    pdf.cell(0, 10, f"DETECTED CONDITION: {medical_info['name']} ({label})", 0, 1)
    
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.cell(0, 8, f"AI Confidence Score: {conf:.1%}", 0, 1)
    pdf.ln(2)
    
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"Pathology Description: {medical_info['desc']}")
    pdf.ln(5)

    # --- 3. VISUAL ANALYSIS ---
    pdf.section_title("VISUAL TELEMETRY ANALYSIS")
    y_start = pdf.get_y()
    
    # Left Image (Original)
    pdf.set_xy(10, y_start + 2)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(90, 5, "Original OCT Scan", 0, 1, 'C')
    if img_path:
        pdf.image(img_path, x=15, y=y_start + 8, w=80, h=80)
        
    # Right Image (Grad-CAM)
    pdf.set_xy(105, y_start + 2)
    pdf.cell(90, 5, "AI Attention Map (Grad-CAM)", 0, 1, 'C')
    if heatmap_path:
        pdf.image(heatmap_path, x=110, y=y_start + 8, w=80, h=80)
    
    # Move cursor down
    pdf.set_y(y_start + 95)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 5, "Note: The Attention Map highlights regions (in red/yellow) that most influenced the AI's diagnostic decision.")
    pdf.ln(8)

    # --- 4. CLINICAL RECOMMENDATIONS ---
    pdf.section_title("CLINICAL PROTOCOLS")
    pdf.set_font("Arial", "B", 11)
    
    # [FIXED] Increased width from 40 to 60 to prevent text overlap
    pdf.cell(60, 8, "Recommended Treatment:", 0, 0)
    
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, medical_info['treatment'])
    
    pdf.set_font("Arial", "B", 11)
    
    # [FIXED] Increased width from 40 to 60 here as well
    pdf.cell(60, 8, "Immediate Action:", 0, 0)
    
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, medical_info['action'])
    pdf.ln(5)

    # --- 5. PHYSICIAN NOTES ---
    if notes:
        pdf.section_title("PHYSICIAN NOTES")
        pdf.body_text(notes)
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')

# --- MAIN UI ---
def main():
    init_db()

    # Session State
    if 'auth' not in st.session_state: st.session_state.auth = False
    if 'messages' not in st.session_state: st.session_state.messages = []

    # LOGIN SCREEN
    if not st.session_state.auth:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.title("🔐 Access Portal")
            st.markdown("### Retinal Diagnostic System")
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Authenticate System"):
                if u == "doctor" and p == "doc123":
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("Access Denied")
        return

    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.title("👁️ NeuroRetina")
        selected = option_menu(
            menu_title=None,
            options=["Analysis Studio", "Patient Registry", "AI Consultant"],
            icons=["activity", "database", "robot"],
            default_index=0,
            styles={
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "5px"},
                "nav-link-selected": {"background-color": "#1976d2"},
            }
        )
        st.divider()
        if st.button("End Session"):
            st.session_state.auth = False
            st.rerun()

    model = load_clinical_model()

    # --- TAB 1: ANALYSIS STUDIO ---
    if selected == "Analysis Studio":
        st.markdown("## 🧬 Clinical Analysis Studio")

        # Top Row: Inputs
        with st.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                pid = st.text_input("Patient ID / MRN", placeholder="Ex: PT-4920")
            with c2:
                file = st.file_uploader("Upload OCT Scan", type=['jpg', 'png', 'jpeg'])
            with c3:
                date_display = st.date_input("Scan Date", datetime.now())

        if file and pid:
            # Layout: Image Left, Results Right
            col_img, col_res = st.columns([1, 1])

            # Save uploaded file temporarily for PDF use
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            img = cv2.imread(tmp_path)
            
            # --- NEW VALIDATION CHECK ---
            is_valid, reason = validate_oct_image(img)

            if not is_valid:
                # Show Error and Stop
                st.error(f"🚫 Invalid Image Detected: {reason}")
                st.image(img, caption="Rejected Image", width=300)
                st.warning("Please upload a valid OCT Retinal Scan (Grayscale).")
            
            else:
                # Image is valid, proceed with AI analysis
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_res = cv2.resize(img_rgb, (224, 224))
                    img_batch = np.expand_dims(img_res, axis=0)

                    # Prediction Logic
                    if model:
                        preds = model.predict(img_batch, verbose=0)
                        idx = np.argmax(preds)
                        label = CLASS_NAMES[idx]
                        conf = float(np.max(preds))
                    else:
                        label = "DEMO_MODE"
                        conf = 0.0
                        preds = np.zeros((1,8))

                    # --- CONFIDENCE THRESHOLD CHECK ---
                    if conf < 0.70:
                        st.error("⚠️ Model Uncertain (Low Confidence)")
                        st.write(f"The AI is only **{conf:.1%}** confident. This does not look like a recognizable retinal disease pattern.")
                        st.image(img_rgb, caption="Unrecognizable Scan", width=300)
                    
                    else:
                        # --- PROCEED WITH VALID RESULTS ---
                        info = MEDICAL_DB.get(label, MEDICAL_DB["NORMAL"])

                        # Heatmap Generation
                        overlay_data = None
                        heatmap_path = None 
                        
                        if model:
                            heatmap = generate_heatmap(img_batch, model)
                            if heatmap is not None:
                                hm_res = cv2.resize(heatmap, (224, 224))
                                hm_u8 = np.uint8(255 * hm_res)
                                jet = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
                                jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
                                overlay_data = cv2.addWeighted(img_res, 0.6, jet, 0.4, 0)

                                overlay_bgr = cv2.cvtColor(overlay_data, cv2.COLOR_RGB2BGR)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as hm_tmp:
                                    cv2.imwrite(hm_tmp.name, overlay_bgr)
                                    heatmap_path = hm_tmp.name

                        with col_img:
                            st.markdown("### Visual Telemetry")
                            tab_a, tab_b = st.tabs(["Raw Scan", "AI Attention Map"])
                            with tab_a: st.image(img_rgb, use_container_width=True, caption="Original Input")
                            with tab_b:
                                if overlay_data is not None: st.image(overlay_data, use_container_width=True, caption="Gradient Class Activation")
                                else: st.warning("Heatmap unavailable.")

                        with col_res:
                            st.markdown("### Diagnostic Output")
                            m1, m2 = st.columns(2)
                            m1.metric("Prediction", label)
                            m2.metric("Confidence", f"{conf:.2%}", delta_color="normal")

                            probs = preds[0]
                            df_probs = pd.DataFrame({"Condition": CLASS_NAMES, "Probability": probs})
                            fig = px.bar(df_probs, x='Probability', y='Condition', orientation='h',
                                         title="Differential Diagnosis", text_auto='.2%',
                                         color='Probability', color_continuous_scale='Blues')
                            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig, use_container_width=True)

                            st.info(f"**Protocol:** {info['treatment']}\n\n**Action:** {info['action']}")

                            notes = st.text_area("Physician Notes")
                            b1, b2 = st.columns(2)
                            if b1.button("💾 Save Record", use_container_width=True):
                                with DB_ENGINE.connect() as conn:
                                    conn.execute(text("INSERT INTO cases (patient_id, date, prediction, confidence, notes) VALUES (:p, :d, :pr, :c, :n)"),
                                                 {"p": pid, "d": str(date_display), "pr": label, "c": conf, "n": notes})
                                    conn.commit()
                                st.toast("Record successfully archived.", icon="✅")

                            pdf_bytes = generate_pdf(pid, label, conf, notes, tmp_path, heatmap_path, info)
                            b2.download_button("📄 Export Professional Report", pdf_bytes, f"Report_{pid}.pdf", "application/pdf", use_container_width=True)
        
        elif not model:
            st.warning("⚠️ Model file 'Trained_Model.keras' not found. App running in restricted mode.")
        else:
            st.info("Awaiting visual input to begin analysis...")

    # --- TAB 2: PATIENT REGISTRY ---
    elif selected == "Patient Registry":
        st.markdown("## 📂 Patient Registry")
        try:
            df = pd.read_sql("SELECT * FROM cases ORDER BY id DESC", DB_ENGINE)

            if not df.empty:
                k1, k2, k3 = st.columns(3)
                k1.metric("Total Scans", len(df))
                k2.metric("High Risk Cases", len(df[df['confidence'] > 0.9]))
                top_disease = df['prediction'].mode()[0]
                k3.metric("Prevalent Condition", top_disease)

                st.divider()
                st.dataframe(
                    df,
                    column_config={
                        "confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
                        "date": "Scan Date"
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No records found in database.")
        except Exception as e:
            st.error(f"Database Connection Error: {e}")

    # --- TAB 3: AI CONSULTANT ---
    elif selected == "AI Consultant":
        st.markdown("## 🤖 Medical Assistant")
        st.caption("Ask questions about protocols, symptoms, or current guidelines.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about a retinal condition..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = "I can answer questions about retinal diseases."
            prompt_lower = prompt.lower()

            found = False
            for key, info in MEDICAL_DB.items():
                if key.lower() in prompt_lower or info['name'].lower() in prompt_lower:
                    response = f"**{info['name']} ({key})**\n\n{info['desc']}\n\n**Treatment:** {info['treatment']}\n**Action:** {info['action']}"
                    found = True
                    break

            if not found:
                response = "I am trained on specific retinal pathologies (AMD, CNV, DME, etc.). Please specify a condition."

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if __name__ == "__main__":
    main()