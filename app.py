import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
  <style>
    body, .main {
        background-color: #121212;
        color: #E0E0E0;
    }

    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    .title {
        color: #E0E0E0;
        text-align: center;
        margin-bottom: 30px;
    }

    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #1e1e1e;
        box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        border: 1px solid #333;
    }

    .malignant {
        color: #ff4c4c;
        font-weight: bold;
    }

    .benign {
        color: #2ecc71;
        font-weight: bold;
    }

    .webcam-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }

    .upload-box {
        border: 2px dashed #1E90FF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        background-color: #1a1a1a;
        color: #E0E0E0;
    }

    .result-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #ffffff;
    }

    .lesion-info {
        margin: 15px 0;
        padding: 15px;
        border-left: 4px solid #1E90FF;
        background-color: #2a2a2a;
        border-radius: 0 8px 8px 0;
        color: #E0E0E0;
    }

    .confidence-bar {
        height: 6px;
        background-color: #333;
        border-radius: 3px;
        margin: 8px 0;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background-color: #1E90FF;
        border-radius: 3px;
    }
</style>

    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def classify_image(image, model):
    try:
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        if image_np.shape[-1] == 4:  # RGBA image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(image_np)
        
        # Get the classification result (assuming single class output)
        if hasattr(results[0], 'probs'):
            probs = results[0].probs
            pred_class = probs.top1
            confidence = probs.top1conf.item()
            class_name = results[0].names[pred_class]
            
            # Create labeled image
            output_image = image_np.copy()
            height, width = output_image.shape[:2]
            
            # Put label at the top center of the image
            label_text = f"{class_name.upper()} ({confidence:.2f})"
            font_scale = min(width, height) / 800
            thickness = max(1, int(min(width, height) / 300))
            
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = (width - text_width) // 2
            text_y = text_height + 20
            
            color = (0, 0, 255) if class_name == "malignant" else (0, 255, 0)
            
            # Draw background rectangle for text
            cv2.rectangle(output_image, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + 10),
                         (255, 255, 255), -1)
            
            # Put text
            cv2.putText(output_image, label_text, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       color, thickness)
            
            # Convert back to RGB for display
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            return {
                'class': class_name,
                'confidence': confidence,
                'labeled_image': Image.fromarray(output_image)
            }
        
        return None
    
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

def main():
    st.title("🩺 Skin Lesion Classifier")
    st.markdown("""
    This application uses a YOLOv8 classification model to analyze skin lesions and predict whether they are malignant.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Webcam Capture", "About"])
    
    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image",use_container_width=True)
            
            if st.button("Classify", key="classify_btn"):
                with st.spinner("Analyzing image..."):
                    result = classify_image(image, model)
                
                if result:
                    with col2:
                        st.image(result['labeled_image'], caption="Classification Result",use_container_width=True)
                    
                    st.markdown("""
                    <div class="result-box">
                        <div class="result-header">Classification Result</div>
                        <div class="lesion-info">
                            <p><b>Prediction:</b> <span class="{class_name}">{class_name}</span></p>
                            <p><b>Confidence:</b> {confidence:.2%}</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {percent}%"></div>
                            </div>
                        </div>
                    </div>
                    """.format(
                        class_name=result['class'],
                        confidence=result['confidence'],
                        percent=result['confidence'] * 100
                    ), unsafe_allow_html=True)
    
    with tab2:
        st.header("Real-time Webcam Classification")
        st.info("Click the 'Start' button below to activate your webcam. Position the skin lesion in the frame.")
        
        # WebRTC streamer for webcam
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Perform classification every 5 frames to reduce computation
            if hasattr(video_frame_callback, "frame_count"):
                video_frame_callback.frame_count += 1
            else:
                video_frame_callback.frame_count = 0
            
            if video_frame_callback.frame_count % 5 == 0:
                # Convert to PIL Image for consistency
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                result = classify_image(pil_img, model)
                if result:
                    video_frame_callback.last_result = result
            
            # Get the last result
            result = getattr(video_frame_callback, "last_result", None)
            
            if result:
                # Convert labeled image back to array
                labeled_img = np.array(result['labeled_image'])
                labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR)
                return labeled_img
            
            return img
        
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing and hasattr(video_frame_callback, "last_result"):
            result = video_frame_callback.last_result
            st.markdown("""
            <div class="result-box">
                <div class="result-header">Classification Result</div>
                <div class="lesion-info">
                    <p><b>Prediction:</b> <span class="{class_name}">{class_name}</span></p>
                    <p><b>Confidence:</b> {confidence:.2%}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {percent}%"></div>
                    </div>
                </div>
            </div>
            """.format(
                class_name=result['class'],
                confidence=result['confidence'],
                percent=result['confidence'] * 100
            ), unsafe_allow_html=True)
    
    with tab3:
        st.header("About This Project")
        st.markdown("""
        ### Skin Lesion Classifier using YOLOv8
        
        This application uses a YOLOv8 classification model trained to identify malignant skin lesions.
        
        **Features:**
        - Single class classification (malignant/benign)
        - Clear visualization of results on the image
        - Confidence percentage with visual indicator
        - Webcam and image upload functionality
        
        **How to Use:**
        1. Upload an image or use the webcam
        2. Click "Classify"
        3. View the results with confidence level
        
        **Disclaimer:**
        This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis.
        """)

if __name__ == "__main__":
    main()