"""
VocalHands - Sign Language Detection Web App
=============================================
Real-time ASL sign language detection using Streamlit.

Deploy on Streamlit Cloud:
1. Push to GitHub
2. Connect repo to streamlit.io/cloud
3. Deploy!
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from PIL import Image
from collections import deque

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="VocalHands - Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .detection-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .detected-sign {
        font-size: 6rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sentence-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        font-size: 1.8rem;
        font-family: 'Courier New', monospace;
        min-height: 80px;
        margin: 1rem 0;
        text-align: center;
        color: #333;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .status-detected {
        background: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    
    .status-waiting {
        background: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    .info-card {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_PATH = "models/knn_sign_model.pkl"

# =============================================================================
# HAND DETECTOR CLASS
# =============================================================================
@st.cache_resource
def get_hand_detector():
    """Initialize MediaPipe hand detector."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return hands, mp_hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles

def extract_landmarks(results):
    """Extract normalized landmarks from MediaPipe results."""
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    landmarks = np.array(landmarks)
    
    # Normalize - center around wrist and scale by palm size
    coords = landmarks.reshape(-1, 3)
    wrist = coords[0].copy()
    coords_centered = coords - wrist
    palm_size = np.linalg.norm(coords_centered[9])
    if palm_size > 0:
        coords_normalized = coords_centered / palm_size
    else:
        coords_normalized = coords_centered
    
    return coords_normalized.flatten()

def draw_landmarks(image, results, mp_hands, mp_drawing, mp_drawing_styles):
    """Draw hand landmarks on image."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return image

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained KNN model."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    return (
        model_data["model"],
        model_data["scaler"],
        model_data["label_encoder"]
    )

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_sign(landmarks, model, scaler, label_encoder):
    """Predict sign from landmarks."""
    features = scaler.transform(landmarks.reshape(1, -1))
    prediction_idx = model.predict(features)[0]
    
    # Get confidence from distances
    distances, _ = model.kneighbors(features)
    mean_distance = np.mean(distances[0])
    confidence = 1 / (1 + mean_distance)
    
    predicted_sign = label_encoder.inverse_transform([prediction_idx])[0]
    return predicted_sign, confidence

# =============================================================================
# SESSION STATE
# =============================================================================
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=10)
if 'last_added' not in st.session_state:
    st.session_state.last_added = ""

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 VocalHands</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time ASL Sign Language Detection | Powered by MediaPipe & KNN</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_encoder = load_model()
    
    if model is None:
        st.error("⚠️ Model file not found!")
        st.info("""
        **To deploy this app:**
        1. Make sure `models/knn_sign_model.pkl` exists in your repository
        2. Train the model locally first: `python train_model.py`
        3. Commit and push the model file to GitHub
        """)
        return
    
    # Get hand detector
    hands, mp_hands, mp_drawing, mp_drawing_styles = get_hand_detector()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05
        )
        
        auto_add = st.checkbox("Auto-add to sentence", value=True)
        
        st.divider()
        
        st.header("📝 Your Sentence")
        st.markdown(
            f'<div class="sentence-box">{st.session_state.sentence or "..."}</div>',
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️", help="Clear"):
                st.session_state.sentence = ""
                st.session_state.last_added = ""
                st.rerun()
        with col2:
            if st.button("⎵", help="Space"):
                st.session_state.sentence += " "
                st.rerun()
        with col3:
            if st.button("⌫", help="Backspace"):
                if st.session_state.sentence:
                    st.session_state.sentence = st.session_state.sentence[:-1]
                    st.rerun()
        
        st.divider()
        
        st.header("📋 Signs Available")
        signs = list(label_encoder.classes_)
        st.caption(f"{len(signs)} signs: {', '.join(signs)}")
        
        st.divider()
        
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. 📸 **Take a photo** of your hand sign
        2. 🔍 The AI will **detect the sign**
        3. ➕ Click **Add to Sentence** or enable auto-add
        4. 📝 Build your message!
        """)
    
    # Main content
    col_camera, col_result = st.columns([1.5, 1])
    
    with col_camera:
        st.subheader("📸 Capture Your Sign")
        
        # Camera input
        camera_image = st.camera_input(
            "Show your ASL sign to the camera",
            key="camera",
            help="Position your hand clearly in the frame"
        )
    
    with col_result:
        st.subheader("🎯 Detection Result")
        
        if camera_image is not None:
            # Process image
            image = Image.open(camera_image)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Detect hands
            results = hands.process(image_np)
            
            # Draw landmarks
            image_with_landmarks = draw_landmarks(
                image_np.copy(), results, mp_hands, mp_drawing, mp_drawing_styles
            )
            
            if results.multi_hand_landmarks:
                st.markdown(
                    '<div class="status-box status-detected">✅ Hand Detected!</div>',
                    unsafe_allow_html=True
                )
                
                # Extract landmarks and predict
                landmarks = extract_landmarks(results)
                
                if landmarks is not None:
                    prediction, confidence = predict_sign(
                        landmarks, model, scaler, label_encoder
                    )
                    
                    if confidence >= confidence_threshold:
                        # Display prediction
                        st.markdown(
                            f'<div class="detection-box">'
                            f'<p class="detected-sign">{prediction}</p>'
                            f'<p style="margin-top: 10px; font-size: 1.2rem;">Confidence: {confidence*100:.1f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Progress bar
                        st.progress(confidence, text=f"Confidence: {confidence*100:.0f}%")
                        
                        # Add to sentence button
                        if auto_add and prediction != st.session_state.last_added:
                            st.session_state.sentence += prediction
                            st.session_state.last_added = prediction
                            st.success(f"Added '{prediction}' to sentence!")
                        elif not auto_add:
                            if st.button(f"➕ Add '{prediction}' to Sentence", type="primary"):
                                st.session_state.sentence += prediction
                                st.rerun()
                    else:
                        st.warning(f"Low confidence ({confidence*100:.0f}%). Try again with clearer hand position.")
            else:
                st.markdown(
                    '<div class="status-box status-waiting">👋 No hand detected. Show your sign!</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="detection-box"><p class="detected-sign">?</p></div>',
                    unsafe_allow_html=True
                )
            
            # Show processed image
            st.image(image_with_landmarks, caption="Processed Image with Landmarks", use_container_width=True)
        else:
            st.markdown(
                '<div class="status-box status-waiting">📷 Take a photo to detect signs</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="detection-box"><p class="detected-sign">-</p></div>',
                unsafe_allow_html=True
            )
    
    # Bottom sentence display
    st.divider()
    st.subheader("💬 Your Message")
    st.markdown(
        f'<div class="sentence-box" style="font-size: 2.5rem;">{st.session_state.sentence or "Start signing to build your message..."}</div>',
        unsafe_allow_html=True
    )
    
    # Quick add buttons
    st.caption("Quick Add:")
    cols = st.columns(10)
    common_signs = ['A', 'B', 'C', 'H', 'I', 'L', 'O', 'V', 'Y', 'SPACE']
    for i, sign in enumerate(common_signs):
        with cols[i]:
            if st.button(sign, key=f"quick_{sign}"):
                if sign == 'SPACE':
                    st.session_state.sentence += " "
                else:
                    st.session_state.sentence += sign
                st.rerun()


if __name__ == "__main__":
    main()
