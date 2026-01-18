import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import pandas as pd  


def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


st.set_page_config(
    page_title="Traffic Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    try:
        return YOLO('last.pt')
    except:
        return None

model = load_model()


banner_base64 = ""
try:
    
    banner_image = Image.open('image.jpg') 
    banner_base64 = get_image_base64(banner_image)
except:
    pass


st.markdown(f"""
    <style>
    .block-container {{
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }}
    
    header {{
        background-color: transparent !important;
    }}

    [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}
    
    [data-testid="stSidebar"] {{
        min-width: 300px !important;
    }}

    .banner-container {{
        position: relative;
        width: 100%; 
        height: 350px;
        background-image: url('data:image/jpeg;base64,{banner_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-color: #0F172A; /* Resim yoksa koyu arka plan */
    }}

    .banner-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 90%;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.9);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}

    .main-content {{
        padding: 2rem;
    }}

    [data-testid="stFileUploader"] {{
        padding: 2rem;
        border: 1px dashed rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        background-color: transparent;
    }}
    
    [data-testid="stMetric"] {{
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        text-align: center;
    }}
    [data-testid="stMetricLabel"] {{ color: #cbd5e1; }}
    [data-testid="stMetricValue"] {{ color: #38bdf8; }}
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    st.divider()
    st.caption("UA-DETRAC / YOLOv8")


st.markdown(
    """
    <div class="banner-container">
        <div class="banner-text">TRAFFIC ANALYSIS PLATFORM</div>
    </div>
    """, 
    unsafe_allow_html=True
)


with st.container():
    col_l, col_m, col_r = st.columns([1, 10, 1])
    
    with col_m:
        st.write("")
        
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown("---")
            
            with st.spinner('AI is analyzing...'):
                if model:
                    
                    results = model.predict(image, conf=conf_threshold)
                    
                    
                    res_plotted = results[0].plot()
                    res_image = res_plotted[..., ::-1]

                    # 1. Görselleri Yan Yana Göster
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(image, caption="Original", use_container_width=True)
                    with c2:
                        st.image(res_image, caption="Detected", use_container_width=True)
                    
                    # 2. İstatistikler (Metrics)
                    st.markdown("### Statistics")
                    boxes = results[0].boxes
                    
                    if len(boxes) > 0:
                        class_names = model.names
                        detected_classes = [class_names[int(cls)] for cls in boxes.cls]
                        counts = {item: detected_classes.count(item) for item in set(detected_classes)}
                        
                        cols = st.columns(len(counts) + 1)
                        cols[0].metric("Total", sum(counts.values()))
                        for idx, (k, v) in enumerate(counts.items()):
                            cols[idx+1].metric(k.upper(), v)
                        
                
                        st.markdown("### Detailed Detection Data")
                        
                        data = []
                        for box in boxes:
                            
                            cls_name = class_names[int(box.cls[0])]
                            conf = float(box.conf[0])
                            coords = box.xyxy[0].tolist() 
                            
                            
                            data.append({
                                "Vehicle Type": cls_name.upper(),
                                "Confidence Score": f"%{conf*100:.1f}",
                                "Coordinates (XYXY)": [int(x) for x in coords]
                            })
                        
                        
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        

                    else:
                        st.warning("No vehicle detected.")
                else:
                    st.error("Model could not be loaded.")
        else:
             st.markdown(
                """
                <div style="text-align:center; margin-top: 20px; color: #64748B;">
                    <p>Upload a traffic image above to get started.</p>
                </div>
                """, unsafe_allow_html=True
            )