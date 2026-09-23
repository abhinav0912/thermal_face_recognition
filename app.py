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

Layout follows the "Emberwatch" mockup: a live status bar (brand + camera/
device info), a two-column main area (live feed | settings + roster), and a
full-width enroll panel -- no native st.sidebar, since the mockup treats
settings/roster as ordinary panels in the main flow, not a collapsible rail.
Visual style is themed via .streamlit/config.toml (native widget colors --
buttons, sliders, toggles) plus the CSS block below (fonts, panel borders,
roster chips, status bar) for what the theme config can't reach. Panel
borders/fonts target Streamlit's documented data-testid hooks rather than
its internal (version-unstable) generated class names, so they degrade
gracefully -- worst case a border doesn't render on some Streamlit version,
nothing breaks.

The brand/status bar and the "N tracked" badge live inside the live_feed()
fragment (not the static page) so they can update every ~50ms tick along
with the video -- connection state, fps, and tracked count are only known
inside that fragment. Re-rendering the (mostly static) brand text that often
is harmless; Streamlit diffs the DOM, so it's not the frame cost it sounds
like. A true pixel-overlay of "LIVE" / frame-counter text on top of the
video image itself (as in the mockup) isn't done here -- Streamlit doesn't
give a stable hook to nest raw HTML inside a specific st.image call, and
faking it with CSS positioning tricks would be fragile across versions.
That info is shown as a status row above the frame instead.

Expression detection is commented out here too, matching the rest of the
pipeline (not a current focus -- revisit next month).

Usage:
    streamlit run app.py
"""

import time

import cv2
import torch
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
[data-testid="stVerticalBlockBorderWrapper"]{ border-color:var(--ew-border) !important; border-radius:6px !important; }
[data-testid="stButton"] button{ border-radius:4px !important; font-weight:600 !important; }
.ew-topbar{ display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap; padding:14px 20px; margin-bottom:8px; background:var(--ew-surface); border:1px solid var(--ew-border); border-radius:6px; }
.ew-topbar-left{ display:flex; align-items:center; gap:14px; }
.ew-brand-mark{ width:32px; height:32px; border-radius:7px; flex-shrink:0; background:radial-gradient(circle at 35% 30%, #fff6ec 0%, var(--ew-accent) 42%, #c75f26 75%, #6a3312 100%); box-shadow:0 0 0 1px var(--ew-border), inset 0 0 8px rgba(0,0,0,.35); }
.ew-brand-name{ font-family:"Archivo Expanded","Archivo",sans-serif; font-weight:700; font-size:1.3rem; color:var(--ew-text); letter-spacing:.01em; }
.ew-brand-sub{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; color:var(--ew-faint); letter-spacing:.08em; text-transform:uppercase; }
.ew-panel-title-row{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
.ew-panel-title{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ew-faint); }
.ew-count-badge{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; color:var(--ew-faint); background:var(--ew-bg); border:1px solid var(--ew-border); border-radius:10px; padding:1px 8px; }
.ew-status-row{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; font-family:"IBM Plex Mono",monospace; font-size:.76rem; color:var(--ew-muted); }
.ew-status-row b{ color:var(--ew-text); font-weight:500; }
.ew-dot{ width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; }
.ew-dot.live{ background:var(--ew-good); box-shadow:0 0 0 3px var(--ew-good-soft); }
.ew-dot.idle{ background:var(--ew-faint); }
.ew-toggle-caption{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; color:var(--ew-faint); margin-top:-8px; margin-bottom:12px; }
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


def _format_source(source: str) -> str:
    """'rtsp://169.254.0.82:554/avc' -> '169.254.0.82:554 · rtsp/avc'; a bare
    webcam index (e.g. '0') -> 'Webcam 0'. Cosmetic only -- camera.py does its
    own parsing of `source` for the actual connection."""
    if source.isdigit():
        return f"Webcam {source}"
    if "://" in source:
        scheme, rest = source.split("://", 1)
        host_port, _, path = rest.partition("/")
        return f"{host_port} · {scheme}/{path}" if path else f"{host_port} · {scheme}"
    return source


def _gpu_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(0)} · cuda"
    return "CPU"


# ─────────────────────────────── Cached, expensive-to-load resources ─────────

@st.cache_resource
def get_detector():
    return ThermalFaceDetector()


@st.cache_resource(show_spinner="Loading model ...")
def get_recognizer(checkpoint_dir: str, device: str):
    return FaceRecognizer(checkpoint_dir, device=(device or None))


# ─────────────────────────────── Process-wide, shared stateful resources ─────
#
# One physical camera and one real-world scene in front of it -- every browser
# tab/viewer should share the SAME connection and tracking state, not each get
# its own. These used to be st.session_state-scoped (per browser session), which
# meant every open tab silently opened its own RTSP connection and ran its own
# full detect+track+recognize loop -- confirmed directly: two connected browser
# sessions were enough to drop the effective frame rate from ~20fps to ~1fps,
# each stealing GPU/CPU/camera-bandwidth from the other. cache_resource (same
# mechanism already used for the detector/recognizer below) fixes this the same
# way: one instance, shared process-wide, keyed by its arguments.

@st.cache_resource
def get_camera(source: str):
    return ThermalCamera(source=source)


@st.cache_resource
def get_tracker():
    return MultiFaceTracker()


# ─────────────────────────────── Recognition ──────────────────────────────────

def identify_all(recognizer, gallery_data, crops_bgr, unknown_threshold, gallery_threshold):
    """Classify every face crop from this frame in a single batched forward pass
    (see inference.py's FaceRecognizer.predict_batch) instead of one model call per
    face -- with several people in frame this is one GPU call instead of N, and the
    gallery-fallback embedding comes from that same pass instead of a second
    backbone run per low-confidence face. Returns a label per crop, in the same
    order as `crops_bgr`."""
    pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    labels = []
    for result, embedding in recognizer.predict_batch(pil_imgs, top_k=1):
        if result["person_conf"] >= unknown_threshold:
            labels.append((f"{result['person_name']} ({result['person_conf']*100:.0f}%)", True))
            continue
        gallery_name, sim = gallery.match(embedding, gallery_data, threshold=gallery_threshold)
        if gallery_name:
            labels.append((f"{gallery_name} ({sim*100:.0f}% live-match)", True))
        else:
            labels.append(("Unknown", False))
    return labels


def annotate(frame, box, label, known):
    x, y, w, h = box
    color = (95, 217, 138) if known else (232, 79, 79)  # BGR: --ew-good / a red for Unknown
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ─────────────────────────────── Settings + roster (right column) ────────────

def render_settings_panel():
    with st.container(border=True):
        st.markdown('<div class="ew-panel-title">Settings</div>', unsafe_allow_html=True)
        running = st.toggle("Run live feed", value=st.session_state.get("running", False))
        st.markdown('<div class="ew-toggle-caption">streaming @ 20fps target</div>',
                    unsafe_allow_html=True)
        st.session_state.running = running

        source = st.text_input("Camera source (RTSP URL or webcam index)", value=DEFAULT_RTSP_URL)
        checkpoint_dir = st.text_input("Checkpoint directory", value="checkpoints")
        device = st.selectbox("Device", ["Auto", "cuda", "cpu"], index=0)
        device_arg = "" if device == "Auto" else device
        unknown_threshold = st.slider("Unknown threshold (classifier)", 0.0, 1.0, 0.5, 0.05)
        gallery_threshold = st.slider("Gallery match threshold", 0.0, 1.0, 0.6, 0.05)
    return running, source, checkpoint_dir, device_arg, unknown_threshold, gallery_threshold


def render_roster_panel(names: dict, gallery_data: dict):
    with st.container(border=True):
        st.markdown(
            f'<div class="ew-panel-title-row">'
            f'<span class="ew-panel-title">Registered people</span>'
            f'<span class="ew-count-badge">{len(names) + len(gallery_data)}</span>'
            f'</div>', unsafe_allow_html=True)

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

left_col, right_col = st.columns([2, 1], gap="medium")

with right_col:
    running, source, checkpoint_dir, device_arg, unknown_threshold, gallery_threshold = \
        render_settings_panel()

    names = load_names()
    gallery_data = gallery.load_gallery()
    render_roster_panel(names, gallery_data)

try:
    recognizer = get_recognizer(checkpoint_dir, device_arg)
    detector = get_detector()
except FileNotFoundError as e:
    st.error(f"Could not load model: {e}")
    st.stop()


@st.fragment(run_every=0.05)
def live_feed():
    """
    Auto-reruns on its own (every ~50ms) WITHOUT touching the rest of the
    page -- st.fragment patches just this piece of the DOM, unlike an
    st.rerun()-in-a-loop approach, which forces the entire page (settings,
    roster, every widget) to tear down and redraw on every single frame.

    Renders the brand/status bar itself (see module docstring for why) so
    connection state, fps, and tracked count can update every tick even
    while the rest of the page stays untouched.
    """
    running = st.session_state.get("running", False)
    tracked_count = 0
    status_html = (
        f'<span><span class="ew-dot idle"></span>IDLE</span>'
        f'<span>{_format_source(source)}</span>'
        f'<span>{_gpu_label(recognizer.device)}</span>'
    )

    frame_shown = False
    if running:
        try:
            camera = get_camera(source)
        except ConnectionError as e:
            st.error(f"Could not open camera: {e}")
            camera = None

        if camera is not None:
            tracker = get_tracker()
            gallery_data = gallery.load_gallery()

            now = time.time()
            last_tick = st.session_state.get("ew_last_tick")
            fps = (1.0 / (now - last_tick)) if last_tick else 0.0
            st.session_state.ew_last_tick = now

            ok, frame = camera.read()
            if ok and frame is not None:
                tracks = tracker.update(detector.detect(frame))
                tracked_count = len(tracks)

                boxes, crops = [], []
                for box in tracks.values():
                    crop = detector.crop(frame, box)
                    if crop.size == 0:
                        continue
                    boxes.append(box)
                    crops.append(crop)

                labels = identify_all(recognizer, gallery_data, crops,
                                       unknown_threshold, gallery_threshold)
                for box, (label, known) in zip(boxes, labels):
                    annotate(frame, box, label, known)

                status_html = (
                    f'<span><span class="ew-dot live"></span>LIVE</span>'
                    f'<span>{_format_source(source)}</span>'
                    f'<span>{fps:.1f} <b>fps</b></span>'
                    f'<span>{_gpu_label(recognizer.device)}</span>'
                )
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_shown = True
            else:
                status_html = (
                    f'<span><span class="ew-dot idle"></span>RECONNECTING</span>'
                    f'<span>{_format_source(source)}</span>'
                )

    st.markdown(
        f'<div class="ew-topbar">'
        f'<div class="ew-topbar-left">'
        f'<div class="ew-brand-mark"></div>'
        f'<div>'
        f'<div class="ew-brand-name">Thermal Face Recognition</div>'
        f'<div class="ew-brand-sub">Live identification console</div>'
        f'</div></div>'
        f'<div class="ew-status-row">{status_html}</div>'
        f'</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f'<div class="ew-panel-title-row">'
            f'<span class="ew-panel-title">Live feed</span>'
            f'<span class="ew-count-badge">{tracked_count} tracked</span>'
            f'</div>', unsafe_allow_html=True)
        if frame_shown:
            st.image(frame_rgb, channels="RGB")
        elif running:
            st.warning("No frame received, retrying ...")
        else:
            st.info("Live feed paused. Toggle it on above, or enroll someone new below.")


with left_col:
    live_feed()


# ─────────────────────────────── Enroll ────────────────────────────────────────

with st.container(border=True):
    st.markdown('<div class="ew-panel-title">Enroll a new person</div>', unsafe_allow_html=True)
    st.caption("No retraining needed -- captures a few frames on the spot and matches by "
               "similarity. Weaker than a fully trained identity, but recognizable immediately. "
               "Whoever is largest/closest to the camera gets enrolled.")

    enroll_name = st.text_input("Name")
    btn_col, pause_col = st.columns([1, 1])
    capture_clicked = btn_col.button(
        "Capture & enroll", type="primary",
        disabled=not enroll_name.strip() or st.session_state.get("running", False))
    if pause_col.button("Pause feed first", type="secondary",
                         disabled=not st.session_state.get("running", False)):
        st.session_state.running = False
        st.rerun()

    if capture_clicked:
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
