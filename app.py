import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import sys
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set page layout
st.set_page_config(page_title="IoT Intrusion Detection XAI", layout="wide")

# ---------------------------------------------------------
# Configuration & Paths (Requirement 9: Use pathlib)
# ---------------------------------------------------------
# Define paths using pathlib for robust cross-platform compatibility
FL_SYSTEM_DIR = Path('fl_system')
MODEL_PATH = FL_SYSTEM_DIR / 'saved_models' / 'centralized_model.pth'
DATA_DIR = Path('data')
SCALER_PATH = DATA_DIR / 'scaler.joblib'
ENCODER_PATH = DATA_DIR / 'label_encoder.joblib'
TEST_DATA_PATH = DATA_DIR / 'test.csv'

NUM_FEATURES = 46
NUM_CLASSES = 34

# ---------------------------------------------------------
# Initialization & Dependency Checking (Requirement 2, 3, 7, 8)
# ---------------------------------------------------------
# Introduce a global flag for graceful degradation
FL_AVAILABLE = True
missing_resources = []

# Verify all necessary FL resources exist
if not FL_SYSTEM_DIR.exists():
    missing_resources.append("Federated Learning Directory (fl_system/)")
    FL_AVAILABLE = False
if not MODEL_PATH.exists():
    missing_resources.append("Centralized Model (centralized_model.pth)")
    FL_AVAILABLE = False
if not SCALER_PATH.exists():
    missing_resources.append("Scaler (scaler.joblib)")
    FL_AVAILABLE = False
if not ENCODER_PATH.exists():
    missing_resources.append("Label Encoder (label_encoder.joblib)")
    FL_AVAILABLE = False
if not TEST_DATA_PATH.exists():
    missing_resources.append("Test Data (test.csv)")
    FL_AVAILABLE = False

# Only dynamically import model architecture and XAI libraries if FL_AVAILABLE is True (Requirement 1)
if FL_AVAILABLE:
    try:
        # Dynamically import ImprovedTransformer
        # Convert path to absolute string for sys.path
        fl_path_str = str(FL_SYSTEM_DIR.absolute())
        if fl_path_str not in sys.path:
            sys.path.append(fl_path_str)
        
        # Import the model module dynamically
        model_module = importlib.import_module("model")
        ImprovedTransformer = model_module.ImprovedTransformer
        
        # Also import shap and captum which might crash if not installed in basic environment
        import shap
        from captum.attr import IntegratedGradients
        
    except Exception as e:
        missing_resources.append(f"FL Modules Import Error: {e}")
        FL_AVAILABLE = False

# ---------------------------------------------------------
# Helper Functions & Model Loading (Requirement 8)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    """
    Loads machine learning resources (Model, Scaler, Data).
    Only called if FL_AVAILABLE is True to prevent crashes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model (Dynamically instantiated)
    model = ImprovedTransformer(input_dim=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    
    # Load Scaler and Label Encoder
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    
    # Load Background Data for SHAP (taking 256 samples from test set)
    df_bg = pd.read_csv(TEST_DATA_PATH, nrows=256)
    X_bg_raw = df_bg.iloc[:, :-1].values
    X_bg_scaled = scaler.transform(X_bg_raw)
    X_bg_tensor = torch.tensor(X_bg_scaled, dtype=torch.float32).to(device)
    
    return model, scaler, label_encoder, X_bg_tensor, device

# Only attempt to load resources if everything is available
if FL_AVAILABLE:
    try:
        model, scaler, label_encoder, X_bg_tensor, device = load_resources()
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        FL_AVAILABLE = False

# ---------------------------------------------------------
# UI Rendering (Requirement 6)
# ---------------------------------------------------------
st.title("🛡️ IoT Network Intrusion Detection & Explainability")

# Display Streamlit warning instead of crashing (Requirement 4, 7)
if not FL_AVAILABLE:
    st.warning("⚠️ **AI prediction module is unavailable in this deployment.**\n\n"
               "The Federated Learning module is intentionally excluded from GitHub because it is still under development. "
               "Prediction and Explainability features will automatically become available once the local FL module is integrated.")
    with st.expander("View missing resources"):
        for resource in missing_resources:
            st.write(f"- {resource}")

st.markdown("Enter network features manually or load a random sample to see the model's prediction and Explainable AI (XAI) analysis.")

# Session state to hold features
if "feature_values_str" not in st.session_state:
    st.session_state.feature_values_str = ",".join(["0"] * NUM_FEATURES)

def load_random_sample():
    # Defensive programming: ensure test data exists before loading (Requirement 2)
    if not FL_AVAILABLE or not TEST_DATA_PATH.exists():
        st.warning("Cannot load random sample: Test data is unavailable.")
        return
    try:
        # read a quick chunk from the start of the file for fast random sampling
        df_sample = pd.read_csv(TEST_DATA_PATH, nrows=1000)
        if not df_sample.empty:
            random_row = df_sample.sample(1).iloc[0, :-1].values
            st.session_state.feature_values_str = ",".join([str(val) for val in random_row])
    except Exception as e:
        st.error(f"Failed to load sample: {e}")

col1, col2 = st.columns([1, 4])
with col1:
    # Disable button if FL_AVAILABLE is False (Requirement 4)
    st.button("🎲 Load Random Sample", on_click=load_random_sample, disabled=not FL_AVAILABLE)

st.subheader("Network Features Input")

feature_names = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate", 
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", 
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", 
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", 
    "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", 
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"
]

st.markdown("**Expected Format:** Enter a comma-separated list of the 46 numeric values corresponding to the following network features:")
st.info(", ".join(feature_names))

# Text area for CSV input
input_str = st.text_area("Comma-separated values:", value=st.session_state.feature_values_str, height=150)

# Disable the predict button and skip execution if FL is not available (Requirement 4)
if st.button("🚀 Predict & Explain", type="primary", disabled=not FL_AVAILABLE):
    with st.spinner("Analyzing network traffic and generating explanations..."):
        try:
            # Parse input string
            feature_inputs = [float(x.strip()) for x in input_str.split(",") if x.strip()]
            if len(feature_inputs) != NUM_FEATURES:
                st.error(f"Expected {NUM_FEATURES} features, but got {len(feature_inputs)}.")
                st.stop()
        except Exception as e:
            st.error(f"Invalid input format: {e}")
            st.stop()
            
        # 1. Preprocess
        input_array = np.array(feature_inputs).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        input_tensor.requires_grad_()
        
        # 2. Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            
        pred_class_name = label_encoder.inverse_transform([pred_idx])[0]
        
        st.markdown("---")
        st.subheader("🔮 Prediction Results")
        
        # Display Prediction
        if str(pred_class_name).lower() in ["normal", "benign", "0", 0]:
            st.success(f"**Status:** {pred_class_name} (Confidence: {confidence:.2%})")
        else:
            st.error(f"**Alert! Attack Detected:** {pred_class_name} (Confidence: {confidence:.2%})")
            
        # 3. Explainability (Requirement 4: skip if unavailable, handled by disabled predict button)
        st.markdown("---")
        st.subheader("🧠 Explainable AI Insights")
        
        tab1, tab2 = st.tabs(["SHAP (Feature Importance)", "Captum (Integrated Gradients)"])
        
        with tab1:
            st.markdown("**SHAP (SHapley Additive exPlanations)** shows how each feature contributed to pushing the model output from the base value to the current prediction.")
            try:
                # DeepExplainer requires evaluating gradients so we don't use torch.no_grad()
                explainer = shap.DeepExplainer(model, X_bg_tensor)
                shap_values = explainer.shap_values(input_tensor, check_additivity=False)
                
                if isinstance(shap_values, list):
                    shap_values_to_plot = shap_values[pred_idx]
                    if shap_values_to_plot.ndim > 1:
                        shap_values_to_plot = shap_values_to_plot[0]
                else:
                    if shap_values.ndim == 3:
                        shap_values_to_plot = shap_values[0, :, pred_idx]
                    elif shap_values.ndim == 2:
                        shap_values_to_plot = shap_values[0]
                    else:
                        shap_values_to_plot = shap_values
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                indices = np.argsort(np.abs(shap_values_to_plot))[-15:] # Top 15
                
                colors = ['salmon' if val < 0 else 'skyblue' for val in shap_values_to_plot[indices]]
                ax.barh(np.arange(len(indices)), shap_values_to_plot[indices], color=colors, edgecolor='black')
                ax.set_yticks(np.arange(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices], fontweight='bold')
                ax.set_xlabel('SHAP Value (Impact on Model Output)', weight='bold')
                ax.set_title(f'Top 15 Features Driving Prediction: {pred_class_name} (SHAP)', weight='bold')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP Error: {e}")
                
        with tab2:
            st.markdown("**Captum Integrated Gradients** approximates the integral of gradients along the path from a baseline to the given input.")
            try:
                ig = IntegratedGradients(model)
                attributions, delta = ig.attribute(input_tensor, target=pred_idx, return_convergence_delta=True)
                
                attr_np = attributions.cpu().detach().numpy()[0]
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                indices_ig = np.argsort(np.abs(attr_np))[-15:]
                
                colors_ig = ['salmon' if val < 0 else 'mediumpurple' for val in attr_np[indices_ig]]
                ax2.barh(np.arange(len(indices_ig)), attr_np[indices_ig], color=colors_ig, edgecolor='black')
                ax2.set_yticks(np.arange(len(indices_ig)))
                ax2.set_yticklabels([feature_names[i] for i in indices_ig], fontweight='bold')
                ax2.set_xlabel('Integrated Gradients Attribution', weight='bold')
                ax2.set_title(f'Top 15 Features Driving Prediction: {pred_class_name} (Captum)', weight='bold')
                ax2.grid(axis='x', linestyle='--', alpha=0.7)
                
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Captum Error: {e}")

        # Save context for chatbot
        st.session_state.prediction_context = {
            "prediction": pred_class_name,
            "confidence": confidence,
            "top_shap_features": [feature_names[i] for i in indices][-5:],
            "top_captum_features": [feature_names[i] for i in indices_ig][-5:]
        }

# ---------------------------------------------------------
# Chatbot Interface (Requirement 5)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💬 Explainability Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the AI's decision or cybersecurity..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Setup LLM
                chat_model = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_tokens=512,
                )
                
                # Setup Prompt
                context_str = f"The dataset and dashboard use the following 46 input features: {', '.join(feature_names)}.\n"
                
                # Modify prompt context dynamically based on FL availability
                if "prediction_context" not in st.session_state:
                    if FL_AVAILABLE:
                        context_str += "No prediction has been made yet. Advise the user to click 'Predict & Explain' for model insights."
                    else:
                        context_str += "The prediction module is currently unavailable because the Federated Learning module is missing. The user CANNOT make predictions right now. Advise the user that prediction is disabled, but you can still answer general cybersecurity questions."
                else:
                    ctx = st.session_state.prediction_context
                    context_str += f"\nThe model just predicted the network traffic as '{ctx['prediction']}' with {ctx['confidence']:.2%} confidence.\n"
                    context_str += f"The top features that drove this specific prediction according to SHAP are: {', '.join(ctx['top_shap_features'])}.\n"
                    context_str += f"The top features according to Captum are: {', '.join(ctx['top_captum_features'])}."
                
                template = """You are a helpful AI cybersecurity assistant. A user is interacting with an Explainable AI dashboard for an IoT Intrusion Detection System.
Dashboard & Model Context:
{context}

User Question: {question}

Answer the user's question clearly and concisely. 
- If they ask about the latest prediction, refer to the SHAP/Captum data. 
- If they ask general questions about the input features (e.g., "explain all input features"), explain what those network features represent in cybersecurity generally.
- If they ask a general question, answer it using your background knowledge.
- If prediction is disabled, kindly remind them but answer their general question anyway.
Answer:"""
                
                prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
                chain = prompt_template | chat_model | StrOutputParser()
                
                response = chain.invoke({"context": context_str, "question": prompt})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Chatbot encountered an error: {e}")
