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

Expression detection is commented out here too, matching the rest of the
pipeline (not a current focus -- revisit next month).

Usage:
    streamlit run app.py
"""

import cv2
import streamlit as st
from PIL import Image

import gallery
from camera import ThermalCamera, DEFAULT_RTSP_URL
from face_detector import ThermalFaceDetector, MultiFaceTracker
from inference import FaceRecognizer
from person_names import load_names

st.set_page_config(page_title="Thermal Face Recognition", layout="wide")


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
    st.header("Settings")
    source = st.text_input("Camera source (RTSP URL or webcam index)", value=DEFAULT_RTSP_URL)
    checkpoint_dir = st.text_input("Checkpoint directory", value="checkpoints")
    device = st.selectbox("Device", ["Auto", "cuda", "cpu"], index=0)
    device_arg = "" if device == "Auto" else device
    unknown_threshold = st.slider("Unknown threshold (classifier)", 0.0, 1.0, 0.5, 0.05)
    gallery_threshold = st.slider("Gallery match threshold", 0.0, 1.0, 0.6, 0.05)

    st.divider()
    st.header("Registered people")
    names = load_names()
    gallery_data = gallery.load_gallery()
    if names:
        st.caption("Trained (classifier)")
        for pid, name in sorted(names.items()):
            st.text(f"  {name}  (ID {pid})")
    if gallery_data:
        st.caption("Live-enrolled (gallery)")
        for name in sorted(gallery_data.keys()):
            st.text(f"  {name}")
    if not names and not gallery_data:
        st.caption("No one registered yet.")


# ─────────────────────────────── Main area ────────────────────────────────────

st.title("Thermal Face Recognition")

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
    page -- st.fragment patches just this piece of the DOM, unlike the
    st.rerun()-in-a-loop approach this replaced, which forced the entire
    page (sidebar, title, every widget) to tear down and redraw on every
    single frame. That whole-page redraw is what caused the flicker.
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

    ok, frame = camera.read()
    if ok and frame is not None:
        tracks = tracker.update(detector.detect(frame))
        for box in tracks.values():
            crop = detector.crop(frame, box)
            if crop.size == 0:
                continue
            label = identify(recognizer, gallery_data, crop, unknown_threshold, gallery_threshold)
            annotate(frame, box, label)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    else:
        st.warning("No frame received, retrying ...")


if running:
    st.caption("Uncheck the box above to pause the feed before enrolling someone new.")
live_feed()

if not running:
    st.subheader("Enroll a new person")
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
