import streamlit as st
import os
from PIL import Image
from pathlib import Path
import urllib.request
from io import BytesIO
import numpy as np
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer



# -----------------------------
# REAL-ESRGAN (CLOUD / CPU)
# -----------------------------

MODEL_FILENAME = "RealESRGAN_x4plus.pth"

MODEL_URL = (
    "https://"
    + "github.com/"
    + "sajidumar2525-dev/nexora/releases/latest/download/"
    + MODEL_FILENAME
)

MODEL_DIR = Path.home() / ".cache" / "nexora"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def ensure_model():
    if MODEL_PATH.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with st.spinner("Downloading Nexora AI model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Free CPU hosting has limited RAM. Tiling keeps peak inference memory lower.
TILE_SIZE = 256

@st.cache_resource(show_spinner="Loading Nexora AI model...")
def load_upsampler():
   ensure_model()

    # The x4 model supports arbitrary final output scale through outscale.
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    return RealESRGANer(
        scale=4,
        model_path=str(MODEL_PATH),
        model=model,
        tile=TILE_SIZE,
        tile_pad=10,
        pre_pad=0,
        half=False,  # CPU deployment: use FP32 for compatibility
    )


def enhance_image(image: Image.Image, scale_value: int) -> bytes:
    """Enhance an RGB PIL image and return PNG bytes."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    upsampler = load_upsampler()
    output_bgr, _ = upsampler.enhance(
        bgr,
        outscale=float(scale_value),
    )

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_rgb)

    buffer = BytesIO()
    output_image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# Keep the free CPU instance safe from very large output allocations.
MAX_OUTPUT_PIXELS = 60_000_000

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Nexora",
    page_icon=":material/auto_awesome:",
    layout="centered"
)


# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

/* -----------------------------
   COMPACT UPLOAD AREA
----------------------------- */

[data-testid="stFileUploaderDropzone"] {
    min-height: 88px !important;
    height: 88px !important;
    padding: 12px 16px !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    padding: 0 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] div {
    font-size: 14px !important;
}

[data-testid="stFileUploaderDropzone"] button {
    border-radius: 10px !important;
    padding: 6px 18px !important;
    font-size: 14px !important;
}

    /* --------------------------------
       GLOBAL LAYOUT
    -------------------------------- */

    .block-container {
        max-width: 900px;
        padding-top: 3.5rem;
        padding-bottom: 3rem;
    }

/* --------------------------------
   NEXORA BRAND
-------------------------------- */

.brand-logo {
    display: flex;
    justify-content: center;
    margin-bottom: 10px;
}

.brand-logo img {
    width: 82px;
    height: 82px;
    object-fit: contain;
}

.brand-name {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #22d3ee, #6366f1, #d946ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-tagline {
    text-align: center;
    color: #888888;
    font-size: 14px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 38px;
}
    
    /* --------------------------------
       MAIN TITLE
    -------------------------------- */

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 750;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 42px;
    }


    /* --------------------------------
       SECTION TITLES
    -------------------------------- */

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 28px;
        margin-bottom: 12px;
    }


    /* --------------------------------
       UPLOAD AREA
    -------------------------------- */

    [data-testid="stFileUploader"] {
    border: 1px dashed #555555;
    border-radius: 14px;
    padding: 10px;
        transition: border-color 0.2s ease,
                    background-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #888888;
    }


    /* --------------------------------
       BUTTONS
    -------------------------------- */

    div.stButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 11px;
        font-size: 16px;
        font-weight: 650;
        border: 1px solid rgba(128,128,128,0.35);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
    }

    div.stDownloadButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 11px;
        font-size: 16px;
        font-weight: 650;
        border: 1px solid rgba(128,128,128,0.35);
    }


    /* --------------------------------
       RADIO BUTTONS
    -------------------------------- */

    div[role="radiogroup"] {
    gap: 10px;
    margin-bottom: 18px;
}

div[role="radiogroup"] label {
    border: 1px solid rgba(128, 128, 128, 0.30);
    border-radius: 10px;
    padding: 9px 20px;
    transition: all 0.15s ease;
}

div[role="radiogroup"] label:hover {
    border-color: rgba(128, 128, 128, 0.65);
}


    /* --------------------------------
       IMAGE PREVIEW
    -------------------------------- */

    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* --------------------------------
       STATUS
    -------------------------------- */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* --------------------------------
       FOOTER
    -------------------------------- */

    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 24px;
    }
/* -----------------------------
   ENHANCEMENT CONTROL CARD
----------------------------- */

.enhance-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 14px;
    padding: 18px 20px 8px 20px;
    margin-top: 24px;
    margin-bottom: 14px;
}

.control-title {
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 3px;
}

.control-subtitle {
    color: #888888;
    font-size: 13px;
    margin-bottom: 10px;
}

.control-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
}

.control-subtitle {
    color: #888;
    font-size: 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------


# -----------------------------
# NEXORA HEADER
# -----------------------------

logo_path = Path(__file__).parent / "nexora_logo.png"

if logo_path.exists():
    st.image(str(logo_path), width=110)
st.markdown(
    '<div class="brand-name">Nexora</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand-tagline">AI IMAGE ENHANCER</div>',
    unsafe_allow_html=True
)


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP.",
    label_visibility="collapsed"
)

# FILE SIZE VALIDATION
if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > 200:
        st.error("File size must be 200MB or less.")
        uploaded_file = None

# -----------------------------
# IMAGE PREVIEW
# -----------------------------

if uploaded_file is not None:

    st.markdown(
    '<div class="section-title">Original Image</div>',
    unsafe_allow_html=True
)

    st.image(
        uploaded_file,
        width="stretch"
    )

    st.write("")

# -----------------------------
# ENHANCEMENT CONTROLS
# -----------------------------

st.markdown(
    """
    <div class="enhance-card">
        <div class="control-title">Upscale Image</div>
<div class="control-subtitle">
    Choose the enhancement level for your image.
</div>
    </div>
    """,
    unsafe_allow_html=True
)


 # -----------------------------
# UPSCALE CONTROL
# -----------------------------

scale = st.radio(
    "Upscale",
    options=["2x", "3x", "4x"],
    horizontal=True,
    label_visibility="visible",
    key="upscale_scale"
)

scale_value = {
    "2x": 2,
    "3x": 3,
    "4x": 4
}[scale]

# -----------------------------
# ENHANCE BUTTON
# -----------------------------

if st.button(
    ":material/auto_awesome: Enhance Image",
    use_container_width=True,
    disabled=uploaded_file is None
):


    # -----------------------------
    # CHECK UPLOADED IMAGE
    # -----------------------------

    if uploaded_file is None:
        st.warning(":material/upload: Please upload an image first.")
        st.stop()

    # -----------------------------
    # READ UPLOADED IMAGE
    # -----------------------------

    try:
        image = Image.open(uploaded_file)

        # Handle transparent PNG/WebP images
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            background = Image.new("RGB", image.size, "white")

            if image.mode != "RGBA":
                image = image.convert("RGBA")

            background.paste(
                image,
                mask=image.getchannel("A")
            )

            image = background
        else:
            image = image.convert("RGB")

    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        st.stop()

    # -----------------------------
    # RUN AI
    # -----------------------------

    with st.spinner("Enhancing image..."):
        input_pixels = image.width * image.height
        estimated_output_pixels = input_pixels * (scale_value ** 2)

        if estimated_output_pixels > MAX_OUTPUT_PIXELS:
            max_input_pixels = MAX_OUTPUT_PIXELS // (scale_value ** 2)
            st.error(
                f"This image is too large for the free cloud CPU at {scale_value}x. "
                f"Please use an image with about {max_input_pixels / 1_000_000:.1f} MP "
                "or less."
            )
            st.stop()

        try:
            enhanced_bytes = enhance_image(image, scale_value)
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
            st.stop()

    # -----------------------------
    # SUCCESS
    # -----------------------------

    if enhanced_bytes:

        original_image = Image.open(uploaded_file)
        enhanced_image = Image.open(BytesIO(enhanced_bytes))

        st.info(
            f"Original: {original_image.width} × {original_image.height} px  |  "
            f"Enhanced: {enhanced_image.width} × {enhanced_image.height} px"
        )

        st.success(
            ":material/check_circle: Image enhanced successfully!"
        )

        # -----------------------------
        # COMPARISON
        # -----------------------------

        st.markdown(
            '<div class="section-title">Comparison</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.caption("BEFORE")
            st.image(
                uploaded_file,
                width="stretch"
            )

        with col2:
            st.caption(f"AFTER • {scale}")
            st.image(
                enhanced_bytes,
                width="stretch"
            )

        st.write("")

        # -----------------------------
        # DOWNLOAD
        # -----------------------------

        original_name = Path(uploaded_file.name).stem
        download_name = f"{original_name}_{scale}_enhanced.png"

        st.download_button(
            label=":material/download: Download Enhanced Image",
            data=enhanced_bytes,
            file_name=download_name,
            mime="image/png",
            use_container_width=True
        )

    else:
        st.error(":material/error: Something went wrong.")

    
# -----------------------------
# FOOTER
# -----------------------------

st.write("")
st.divider()

st.markdown(
    '<div class="footer">Nexora · Images are processed on the Nexora server.</div>',
    unsafe_allow_html=True
)
