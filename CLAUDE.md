# CLAUDE.md — Thermal Face Recognition

Project memory for Claude. Read this first in any new or compacted session.
**Keep it updated:** at the end of any turn that changes code, decisions, setup or open items, edit the relevant section and add a line to the Changelog.

## Goal
Live person identification from a FLIR A50 thermal camera, for a live demo:
- Show the person's **name** on their face box; show **Unknown** if not recognised.
- Recognise **several people in one frame**.
- Collect face data (video capture) so the model can be trained on new people.
- Add a new person **instantly, without retraining** (gallery fallback), for live enrollment during a demo.
- A browser frontend (Streamlit) styled as the "Emberwatch" mockup. The on-page title must stay **"Thermal Face Recognition"** (Emberwatch was only a design codename).
- Expression/emotion detection is **commented out on purpose** (user will return to it later). Search `expression disabled` to find the pieces.

## Hardware / environment
- Camera: FLIR A50, link-local IP `169.254.0.82` (direct Ethernet to the PC). Working stream: `rtsp://169.254.0.82:554/avc`. Web UI at `http://169.254.0.82`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120). Needs a PyTorch build with CUDA 12.8 (`--index-url https://download.pytorch.org/whl/cu128`); user's venv has torch 2.11.0+cu128.
- User's machine: Windows 11, VS Code, PowerShell, a `.venv`. Project folder: `C:\thermal_face_recognition-main\thermal_face_recognition-main` (double-nested, from a GitHub zip download; an older copy lived at `C:\Users\attemlaa\thermal_face_recognition`).
- Windows gotchas hit so far: `pip`/`python` weren't on PATH (use `py -m pip` / `python -m pip`); Git is installed but needed `C:\Program Files\Git\cmd` added to PATH; `.streamlit` is a dot-folder (create with `mkdir .streamlit` in PowerShell).
- The Claude sandbox cannot reach the camera and has no browser, so camera, WebRTC and UI behaviour can only be verified by the user running things locally. Ask for screenshots/terminal output.

## Code map
| File | Role |
|---|---|
| `model.py` | `DualHeadFaceNet` (MobileNetV2 backbone, `identity_head`, `expression_head`), `get_embedding()` (512-d), shared constants. `expression_head` stays in the architecture so old checkpoints load. |
| `train.py` | Trains identity only (expression loss commented out). Scans `data/thermal-face-128x128/`, rebuilds `checkpoints/label_map.json` each run. Full retrain from pretrained backbone every time. |
| `inference.py` | `FaceRecognizer`: `predict_image`, `embed_image`, name lookup. |
| `camera.py` | `ThermalCamera` (RTSP with reconnect). Forces `cv2.CAP_FFMPEG` for URLs; RTSP transport via env `THERMAL_CAMERA_RTSP_TRANSPORT` (`tcp` default, `udp` to test). |
| `face_detector.py` | `ThermalFaceDetector` (Haar + CLAHE, `min_neighbors=8`, crop with bigger bottom margin 0.45 because the cascade cuts off the chin), `FaceTracker` (one face: smoothing, outlier rejection, recovery after 3 rejections), `MultiFaceTracker` (many faces, proximity matching). |
| `gallery.py` | Instant enrollment: averaged embeddings in `data/gallery.json`, cosine match (default threshold 0.6). |
| `person_names.py` | id → name registry, `data/person_names.json`. |
| `live_inference.py` | OpenCV-window live recognition: classifier first, gallery fallback if confidence < 0.5, else Unknown. `n` enrolls the largest face, `q` quits. |
| `collect_data.py` | Data capture: video mode (preview, SPACE to start, ~4 frames/s sampled into `{id}-TD-A-{n}.jpg`, raw clip in `data/videos/`) and 9 angle stills. Uses `FaceTracker`. Asks for a name on new IDs. |
| `app.py` | Streamlit UI (Emberwatch styling) with live feed over `streamlit-webrtc`. |
| `.streamlit/config.toml` | Dark/amber native theme. Must live in a `.streamlit` folder next to `app.py`. |
| `test_camera_connection.py` | One-off probe for ports/RTSP paths. |
| `benchmark.py`, `ablation_study.py`, `error_analysis.py`, `visualization*.py`, `model_export.py` | Older analysis scripts. They still reference expression and were deliberately left alone. |

## Decisions and gotchas (don't re-learn these)
- `opencv-python` is pinned `<5.0.0`: 5.0 had no `CascadeClassifier`.
- Haar on thermal frames gives false positives (hairline, chest). That's why trackers reject jumps; tracker recovery after repeated rejection exists because rejection-only locked onto a wrong spot forever.
- Closed-set softmax can be confidently wrong on non-face crops, so tracking/consistency matters more than the Unknown threshold alone.
- Gallery matching is weaker than a trained identity; it is a fallback only. Anyone worth keeping should be retrained into the classifier.
- RTSP: decode-corruption spam ("corrupted macroblock") seen with UDP. TCP was forced, then made switchable because TCP stalls on a lossy link could cause a blank/laggy feed. The UDP-vs-TCP A/B result is **not yet reported**. Check cable and NIC speed/duplex if it persists.
- Streamlit: never put a blank line inside the injected `<style>` block (markdown parser breaks it and prints CSS as text). `st.rerun()` loops flicker the whole page; `st.fragment` fixed flicker but was still slower than the native window, hence WebRTC.
- Streamlit's browser path is inherently slower than `live_inference.py`'s native window. The user wants native-like smoothness.
- Training: dataset contents drive everything. Old 113-person data caused heavy class imbalance (new people had 50-100+ frames vs 14).
- Delete bad captured frames by filename; numbering gaps are harmless.

## People
Ids seen in the UI: 120 Abhinav, 121 Chandhini, 122 Jyothsna, 123 Lakshmi, 124 Asher (gallery-only on this branch; the user's local commits reportedly retrain him as a trained person — unverified).

## Git state
- Repo: `abhinav0912/thermal_face_recognition`. My working branch: `claude/thermal-face-recognition-access-hyfolm`. PR #1 already exists for it (do not create another; pushing updates it).
- `origin/main` was committed from the user's local VS Code Claude session. At last comparison, 9 shared `.py` files were byte-identical to my branch; only `app.py` differs (mine has WebRTC). `origin/main` also deleted persons 1-119 and retrained the checkpoint on 120-123.
- The user's local `main` has 2 unpushed commits (`e0fc352` "Fix multi-person detection, GPU pipeline, camera stability; retrain with Asher", `ade84dd` "Remove checkpoints_120plus"). Contents unseen.
- Plan: user pushes `main`; I fetch, read those commits, merge the cleaned dataset and retrained model into my branch, keep my WebRTC `app.py`; then user pulls and tests. Don't tell the user to pull my branch into local `main` before that.
- Commit trailer to use: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01TJZd3mBmQxiwUcHN8fftky`.

## Design (Emberwatch)
Dark warm charcoal `#16130f`, surface `#211c17`, thermal amber accent `#ff7a2f`, good `#5fd98a`, info `#4fc3e8`, alert `#ff5c72`. Fonts: Archivo Expanded (headings), IBM Plex Sans (body), IBM Plex Mono (data). Mockups: https://claude.ai/artifact/Sv5fAR9WoP9gKk26dGSJoi (chosen) and https://claude.ai/artifact/Kn7TvE2UA5dsD1AD5umHML (5 alternatives, not chosen).

## Open items
1. User pushes local `main`; I merge dataset/model work (see Git state).
2. User has not yet run the WebRTC `app.py`; `streamlit-webrtc` + a custom server-side `VideoStreamTrack` is untested. Expect a debugging round (install of `aiortc`/`av`, ICE connection, the `source_video_track` parameter name).
3. `pip` not recognised in the user's venv terminal; use `python -m pip`.
4. Report on UDP vs TCP RTSP.
5. Multi-person live test with two people in frame.
6. Retrain after the dataset cleanup; consider fine-tune-from-checkpoint if retrain time hurts the demo (`--epochs 8 --unfreeze_epoch 999 --batch_size 128` is a fast head-only run).
7. Expression detection to be re-enabled later.

## Changelog
- 2026-10-06: Created this file. State: WebRTC app pushed (`79ff481`), main-vs-branch comparison done, waiting on user's `git push origin main`.
