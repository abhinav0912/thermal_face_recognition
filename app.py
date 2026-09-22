"""
app.py
======
Streamlit frontend for the live thermal face recognition pipeline.

Reuses every existing module as-is (camera.py, face_detector.py,
inference.py, gallery.py, person_names.py) -- this is a UI layer on top of
the same pipeline live_inference.py and collect_data.py already use, not a
reimplementation. Buttons/checkboxes replace the OpenCV-window keypresses
(SPACE/q/n) those scripts relied on, which sidesteps the window-focus
issues that come with driving an OpenCV window's keyboard input directly.

Visual style is themed via .streamlit/config.toml (native widget colors --
buttons, sliders, checkboxes) plus the CSS block below (fonts, panel
borders, roster chips) for what the theme config can't reach. Panel
borders/fonts target Streamlit's documented data-testid hooks rather than
its internal (version-unstable) generated class names, so they degrade
gracefully -- worst case a border doesn't render on some Streamlit version,
nothing breaks.

Expression detection is commented out here too, matching the rest of the
pipeline (not a current focus -- revisit next month).

Usage:
    streamlit run app.py
"""

import time

import cv2
import streamlit as st
from PIL import Image

import gallery
from camera import ThermalCamera, DEFAULT_RTSP_URL
from face_detector import ThermalFaceDetector, MultiFaceTracker
from inference import FaceRecognizer
from person_names import load_names

st.set_page_config(page_title="Thermal Face Recognition", layout="wide")

# NOTE: this whole block must not contain any blank lines. st.markdown()
# runs content through a Markdown parser before letting raw HTML through,
# and a blank line inside a <style> tag makes it treat what follows as a
# new paragraph instead of continuing the style block -- which breaks the
# CSS and dumps the remainder onto the page as literal visible text.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo+Expanded:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{ --ew-bg:#16130f; --ew-surface:#211c17; --ew-border:#3d342a; --ew-text:#f3ece2; --ew-muted:#9c9184; --ew-faint:#6b6357; --ew-accent:#ff7a2f; --ew-accent-soft:rgba(255,122,47,.14); --ew-good:#5fd98a; --ew-good-soft:rgba(95,217,138,.14); --ew-info:#4fc3e8; --ew-info-soft:rgba(79,195,232,.14); }
body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family:"IBM Plex Sans",sans-serif; }
[data-testid="stSidebar"]{ border-right:1px solid var(--ew-border); }
[data-testid="stVerticalBlockBorderWrapper"]{ border-color:var(--ew-border) !important; border-radius:6px !important; }
[data-testid="stButton"] button{ border-radius:4px !important; font-weight:600 !important; }
.ew-topbar{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding:14px 20px; margin-bottom:8px; background:var(--ew-surface); border:1px solid var(--ew-border); border-radius:6px; }
.ew-brand-mark{ width:32px; height:32px; border-radius:7px; flex-shrink:0; background:radial-gradient(circle at 35% 30%, #fff6ec 0%, var(--ew-accent) 42%, #c75f26 75%, #6a3312 100%); box-shadow:0 0 0 1px var(--ew-border), inset 0 0 8px rgba(0,0,0,.35); }
.ew-brand-name{ font-family:"Archivo Expanded","Archivo",sans-serif; font-weight:700; font-size:1.3rem; color:var(--ew-text); letter-spacing:.01em; }
.ew-brand-sub{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; color:var(--ew-faint); letter-spacing:.08em; text-transform:uppercase; }
.ew-panel-title{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ew-faint); margin-bottom:10px; }
.ew-status-row{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; font-family:"IBM Plex Mono",monospace; font-size:.76rem; color:var(--ew-muted); margin-bottom:10px; }
.ew-status-row b{ color:var(--ew-text); font-weight:500; }
.ew-dot{ width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; }
.ew-dot.live{ background:var(--ew-good); box-shadow:0 0 0 3px var(--ew-good-soft); }
.ew-dot.idle{ background:var(--ew-faint); }
.ew-roster-group-label{ font-family:"IBM Plex Mono",monospace; font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; color:var(--ew-faint); margin:12px 0 6px 0; }
.ew-roster-row{ display:flex; align-items:center; gap:9px; padding:4px 0; }
.ew-avatar{ width:22px; height:22px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-family:"IBM Plex Mono",monospace; font-size:.62rem; font-weight:600; color:#16130f; }
.ew-roster-name{ flex:1; font-size:.85rem; color:var(--ew-text); }
.ew-roster-id{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; color:var(--ew-faint); }
.ew-chip{ font-family:"IBM Plex Mono",monospace; font-size:.6rem; letter-spacing:.04em; padding:2px 6px; border-radius:3px; text-transform:uppercase; }
.ew-chip.trained{ background:var(--ew-good-soft); color:var(--ew-good); }
.ew-chip.live{ background:var(--ew-info-soft); color:var(--ew-info); }
</style>
""", unsafe_allow_html=True)

_AVATAR_COLORS = ["#e8b25c", "#8fd0c4", "#e08a7d", "#c7a8e8", "#7ec8e3", "#e8c95c", "#9bd08f"]


def _avatar_color(key: str) -> str:
    return _AVATAR_COLORS[hash(key) % len(_AVATAR_COLORS)]


# ─────────────────────────────── Cached, expensive-to-load resources ─────────

@st.cache_resource
def get_detector():
    return ThermalFaceDetector()


@st.cache_resource(show_spinner="Loading model ...")
def get_recognizer(checkpoint_dir: str, device: str):
    return FaceRecognizer(checkpoint_dir, device=(device or None))


# ─────────────────────────────── Session-scoped, stateful resources ──────────

def get_camera(source: str):
    """(Re)connect only when the source actually changes, not on every rerun."""
    if st.session_state.get("camera") is None or st.session_state.get("camera_source") != source:
        if st.session_state.get("camera") is not None:
            st.session_state.camera.release()
        st.session_state.camera = ThermalCamera(source=source)
        st.session_state.camera_source = source
    return st.session_state.camera


def get_tracker():
    if "tracker" not in st.session_state:
        st.session_state.tracker = MultiFaceTracker()
    return st.session_state.tracker


# ─────────────────────────────── Recognition ──────────────────────────────────

def identify(recognizer, gallery_data, crop_bgr, unknown_threshold, gallery_threshold):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    result = recognizer.predict_image(pil_img, top_k=1)

    if result["person_conf"] >= unknown_threshold:
        return f"{result['person_name']} ({result['person_conf']*100:.0f}%)"

    embedding = recognizer.embed_image(pil_img)
    gallery_name, sim = gallery.match(embedding, gallery_data, threshold=gallery_threshold)
    if gallery_name:
        return f"{gallery_name} ({sim*100:.0f}% live-match)"
    return "Unknown"


def annotate(frame, box, label):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ─────────────────────────────── Sidebar (config + roster) ───────────────────

with st.sidebar:
    st.markdown('<div class="ew-panel-title">Settings</div>', unsafe_allow_html=True)
    source = st.text_input("Camera source (RTSP URL or webcam index)", value=DEFAULT_RTSP_URL)
    checkpoint_dir = st.text_input("Checkpoint directory", value="checkpoints")
    device = st.selectbox("Device", ["Auto", "cuda", "cpu"], index=0)
    device_arg = "" if device == "Auto" else device
    unknown_threshold = st.slider("Unknown threshold (classifier)", 0.0, 1.0, 0.5, 0.05)
    gallery_threshold = st.slider("Gallery match threshold", 0.0, 1.0, 0.6, 0.05)

    st.divider()
    st.markdown('<div class="ew-panel-title">Registered people</div>', unsafe_allow_html=True)
    names = load_names()
    gallery_data = gallery.load_gallery()

    if names:
        st.markdown('<div class="ew-roster-group-label">Trained &middot; classifier</div>',
                    unsafe_allow_html=True)
        for pid, name in sorted(names.items()):
            st.markdown(
                f'<div class="ew-roster-row">'
                f'<span class="ew-avatar" style="background:{_avatar_color(name)};">{name[0].upper()}</span>'
                f'<span class="ew-roster-name">{name}</span>'
                f'<span class="ew-roster-id">ID {pid}</span>'
                f'<span class="ew-chip trained">trained</span>'
                f'</div>', unsafe_allow_html=True)

    if gallery_data:
        st.markdown('<div class="ew-roster-group-label">Live-enrolled &middot; gallery</div>',
                    unsafe_allow_html=True)
        for name in sorted(gallery_data.keys()):
            st.markdown(
                f'<div class="ew-roster-row">'
                f'<span class="ew-avatar" style="background:{_avatar_color(name)};">{name[0].upper()}</span>'
                f'<span class="ew-roster-name">{name}</span>'
                f'<span class="ew-chip live">live</span>'
                f'</div>', unsafe_allow_html=True)

    if not names and not gallery_data:
        st.caption("No one registered yet.")


# ─────────────────────────────── Main area ────────────────────────────────────

st.markdown(
    '<div class="ew-topbar">'
    '<div class="ew-brand-mark"></div>'
    '<div>'
    '<div class="ew-brand-name">Thermal Face Recognition</div>'
    '<div class="ew-brand-sub">Live identification console</div>'
    '</div>'
    '</div>', unsafe_allow_html=True)

try:
    recognizer = get_recognizer(checkpoint_dir, device_arg)
    detector = get_detector()
except FileNotFoundError as e:
    st.error(f"Could not load model: {e}")
    st.stop()

running = st.checkbox("▶ Run live feed", value=st.session_state.get("running", False))
st.session_state.running = running


@st.fragment(run_every=0.05)
def live_feed():
    """
    Auto-reruns on its own (every ~50ms) WITHOUT touching the rest of the
    page -- st.fragment patches just this piece of the DOM, unlike an
    st.rerun()-in-a-loop approach, which forces the entire page (sidebar,
    title, every widget) to tear down and redraw on every single frame.
    """
    if not st.session_state.get("running", False):
        st.info("Live feed paused. Check the box above to start it, "
                 "or enroll someone new below.")
        return

    try:
        camera = get_camera(source)
    except ConnectionError as e:
        st.error(f"Could not open camera: {e}")
        return

    tracker = get_tracker()
    gallery_data = gallery.load_gallery()

    now = time.time()
    last_tick = st.session_state.get("ew_last_tick")
    fps = (1.0 / (now - last_tick)) if last_tick else 0.0
    st.session_state.ew_last_tick = now

    ok, frame = camera.read()
    status = st.empty()
    if ok and frame is not None:
        tracks = tracker.update(detector.detect(frame))
        for box in tracks.values():
            crop = detector.crop(frame, box)
            if crop.size == 0:
                continue
            label = identify(recognizer, gallery_data, crop, unknown_threshold, gallery_threshold)
            annotate(frame, box, label)

        status.markdown(
            f'<div class="ew-status-row">'
            f'<span><span class="ew-dot live"></span>LIVE</span>'
            f'<span>{len(tracks)} <b>tracked</b></span>'
            f'<span>{fps:.1f} <b>fps</b></span>'
            f'</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    else:
        status.markdown(
            '<div class="ew-status-row"><span><span class="ew-dot idle"></span>RECONNECTING</span></div>',
            unsafe_allow_html=True)
        st.warning("No frame received, retrying ...")


with st.container(border=True):
    st.markdown('<div class="ew-panel-title">Live feed</div>', unsafe_allow_html=True)
    if running:
        st.caption("Uncheck the box above to pause the feed before enrolling someone new.")
    live_feed()

if not running:
    with st.container(border=True):
        st.markdown('<div class="ew-panel-title">Enroll a new person</div>', unsafe_allow_html=True)
        st.caption("No retraining needed -- captures a few frames on the spot and matches by "
                   "similarity. Weaker than a fully trained identity, but recognizable immediately. "
                   "Whoever is largest/closest to the camera gets enrolled.")

        enroll_name = st.text_input("Name")
        if st.button("Capture & enroll", disabled=not enroll_name.strip()):
            try:
                camera = get_camera(source)
            except ConnectionError as e:
                st.error(f"Could not open camera: {e}")
                st.stop()

            num_frames = 8
            embeddings = []
            progress = st.progress(0.0, text="Capturing ...")
            attempts, max_attempts = 0, num_frames * 6

            while len(embeddings) < num_frames and attempts < max_attempts:
                ok, frame = camera.read()
                attempts += 1
                if not ok or frame is None:
                    continue
                boxes = detector.detect(frame)
                if not boxes:
                    continue
                crop = detector.crop(frame, boxes[0])
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                embeddings.append(recognizer.embed_image(pil_img))
                progress.progress(len(embeddings) / num_frames,
                                   text=f"Captured {len(embeddings)}/{num_frames}")

            if len(embeddings) < 3:
                st.error("Not enough usable frames -- make sure a face is clearly visible and try again.")
            else:
                gallery.enroll(enroll_name.strip(), embeddings)
                st.success(f"Enrolled '{enroll_name.strip()}' from {len(embeddings)} frames. "
                           f"They should be recognized once you resume the live feed.")
