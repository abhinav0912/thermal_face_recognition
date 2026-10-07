# CLAUDE.md — Thermal Face Recognition

Project memory for Claude. Read this first in any new or compacted session.
**Keep it updated:** at the end of any turn that changes code, decisions, setup or open items, edit the relevant section and add a line to the Changelog.
**Honesty rule (user, 2026-10-07):** in this project always be truthful. If I don't know something, or haven't verified it, say "I do not know" rather than guessing or stating a guess as fact. Label unverified claims as unverified.
**User preference:** when the user asks for "next steps", walk them through each step one at a time with exact PowerShell commands (they are on Windows 11 PowerShell, not Git Bash) and say what they should see after each.
**Before editing code, run `git fetch origin` and read `git log origin/main`:** a second Claude session runs in the user's VS Code and commits straight to `main`. Its code was built against the real camera and is the tested line. Don't overwrite it from the sandbox, which can't reach the camera.

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
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120), driver CUDA 13.2. Needs a PyTorch build from the CUDA 12.8 index. `pyproject.toml` pins torch/torchvision to `https://download.pytorch.org/whl/cu128` for `uv`.
- User's machine: Windows 11, VS Code, PowerShell. Project folder: `C:\thermal_face_recognition-main\thermal_face_recognition-main` (double-nested, from a GitHub zip download). The user installs with **plain `pip`** (told to me 2026-10-07). Earlier I wrongly guessed the venv was uv-based; `main` ships `pyproject.toml`/`uv.lock` (from the user's other Claude session) but the user does not use uv. Verified from the user's terminal 2026-10-07: `.venv` has torch `2.11.0+cu128`, `cuda.is_available()` True, Python 3.14.x, and **no pip module** (`python -m pip` → "No module named pip"), which is what a uv-created venv looks like (not confirmed how it was made). All app dependencies were already installed: `python -m streamlit run app.py` starts, loads the 5-person model on cuda. No install step needed. Plain `pip` was "not recognized" in their terminal at one point, so use `python -m pip`. `requirements.txt` would pull CPU-only torch on a fresh venv; for a fresh install use `--index-url https://download.pytorch.org/whl/cu128` for torch/torchvision.
- The network sits behind a TLS-inspecting corporate proxy (Zscaler); `pyproject.toml` sets `[tool.uv] system-certs = true` for that.
- Windows gotchas: Git is installed but needed `C:\Program Files\Git\cmd` added to PATH; `.streamlit` is a dot-folder.
- The Claude sandbox cannot reach the camera and has no browser. Camera, UI and speed can only be verified by the user running things locally. Ask for screenshots/terminal output.

## Code map (state of `main` at `f655745`)
| File | Role |
|---|---|
| `model.py` | `DualHeadFaceNet` (MobileNetV2, `identity_head`, `expression_head`), `get_embedding()`, `forward_with_embedding()` (one backbone pass gives logits and the gallery embedding). `expression_head` stays so old checkpoints load. |
| `train.py` | Identity-only training (expression loss commented out). Scans `data/thermal-face-128x128/`, rebuilds `checkpoints/label_map.json`. `cudnn.benchmark`, mixed precision (autocast + GradScaler), OS-aware `--workers` (0 on Windows: workers > 0 deadlocked). Full retrain from pretrained backbone each run. |
| `inference.py` | `FaceRecognizer`: `predict_image`, `predict_batch` (one batched GPU pass per frame), `embed_image`, name lookup. |
| `camera.py` | `ThermalCamera`: **background reader thread**, so `.read()` returns the latest frame instantly and never blocks. `cv2.CAP_FFMPEG` for URLs. RTSP transport defaults to **UDP** (env `THERMAL_CAMERA_RTSP_TRANSPORT=tcp` to switch). |
| `face_detector.py` | `ThermalFaceDetector`: Haar + CLAHE, **two passes** (strict, then relaxed fallback for off-angle faces) with a grey-value std texture filter on relaxed boxes, overlap-coefficient dedup, per-pass cascade instances and try/except around `detectMultiScale` (intermittent OpenCV crash). Crop uses a bigger bottom margin (cascade cuts the chin). `FaceTracker` (one face) and `MultiFaceTracker` (many faces, proximity matching, smoothing, outlier rejection, recovery). |
| `gallery.py` | Instant enrollment: averaged embeddings in `data/gallery.json`, cosine match (default 0.6). |
| `person_names.py` | id → name, `data/person_names.json`. |
| `live_inference.py` | OpenCV-window live recognition: classifier first, gallery fallback below 0.5 confidence, else Unknown. `n` enrolls the largest face, `q` quits. |
| `collect_data.py` | Video capture (preview, SPACE to start, ~4 frames/s into `{id}-TD-A-{n}.jpg`, raw clip in `data/videos/`) and angle stills. Asks for a name on new IDs. |
| `app.py` | Streamlit UI in Emberwatch style: two-column layout, no native sidebar, live status bar, `st.toggle` for the feed, `st.fragment(run_every=0.05)` for the live view. Camera and tracker are `st.cache_resource` (several browser tabs used to each open their own RTSP connection and GPU loop). |
| `.streamlit/config.toml` | Dark/amber native theme. |
| `pyproject.toml` | uv project: dependencies, cu128 torch index, system certs. |
| `test_camera_connection.py` | One-off probe for ports/RTSP paths. |
| `benchmark.py`, `ablation_study.py`, `error_analysis.py`, `visualization*.py`, `model_export.py` | Older analysis scripts. |
| `checkpoints/` | Current model: 5 people. `checkpoints_backup_4people_epoch7/` is the previous 4-person model. |

## Decisions and gotchas (don't re-learn these)
- `opencv-python` is pinned `<5.0.0` (5.0 had no `CascadeClassifier`).
- **RTSP: TCP times out opening the stream on this network (30 s FFmpeg timeout); UDP connects immediately.** UDP shows some "corrupted macroblock" log spam; that is harmless noise. Don't force TCP.
- A blocking `camera.read()` froze the whole Streamlit rerun, hence the reader thread.
- Haar on thermal frames gives false positives (hairline, chest) and can return several overlapping boxes for one face, so detection is two-pass with dedup and the trackers reject jumps.
- Closed-set softmax can be confidently wrong on non-face crops. Tracking and consistency matter more than the Unknown threshold alone.
- Gallery matching is weaker than a trained identity. Anyone worth keeping should be retrained into the classifier.
- Streamlit: never put a blank line inside the injected `<style>` block (the markdown parser breaks it and prints CSS as text). `st.rerun()` loops flicker the whole page.
- Streamlit's browser path is inherently slower than `live_inference.py`'s native window; the user wants native-like smoothness. Whether `main`'s current `app.py` is fast enough is **not yet confirmed**.
- **WebRTC option (parked):** an `streamlit-webrtc` version of `app.py` with a custom server-side `VideoStreamTrack` exists in git history at commit `79ff481` (old branch tip). It was never run. Revive it only if `main`'s Streamlit feed is still too laggy. It predates the camera reader thread and the UDP default, so it needs re-basing on those.
- Training data: the old 113-person set was deleted (it caused heavy class imbalance). The roster is whoever has been collected live.
- Delete bad captured frames by filename; numbering gaps are harmless.

## People (all 5 are in the trained classifier)
`data/person_names.json`: 120 Abhinav, 121 Chandhini, 122 Jyothsna, 123 Lakshmi, 124 Asher. Images in `data/thermal-face-128x128/` per id: 120 → 121, 121 → 61, 122 → 113, 123 → 62, 124 → 231. Uneven counts can bias the classifier toward 124.

## Git state
- Repo: `abhinav0912/thermal_face_recognition`. My branch: `claude/thermal-face-recognition-access-hyfolm`. **PR #1** already exists (don't create another; pushing updates it).
- 2026-10-07: the user chose `main` as the base. I merged `origin/main` (`f655745`) into my branch and took main's version of every file, so the branch tree equals `main` plus this updated `CLAUDE.md`. My earlier work stays in history.
- Day to day, commits come from both sessions. Fetch before changing anything; prefer small, additive edits to files `main` has changed.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01TJZd3mBmQxiwUcHN8fftky`.

## Design (Emberwatch)
Dark warm charcoal `#16130f`, surface `#211c17`, thermal amber accent `#ff7a2f`, good `#5fd98a`, info `#4fc3e8`, alert `#ff5c72`. Fonts: Archivo Expanded (headings), IBM Plex Sans (body), IBM Plex Mono (data). Mockups: https://claude.ai/artifact/Sv5fAR9WoP9gKk26dGSJoi (chosen) and https://claude.ai/artifact/Kn7TvE2UA5dsD1AD5umHML (5 alternatives, not chosen).

## Open items
1. User runs `streamlit run app.py` from `main`/this branch and reports speed vs the native window. If still laggy, consider porting the WebRTC approach (see gotchas) or an MJPEG endpoint.
2. Live test with two people in frame (two-pass detection and dedup were added for this).
3. Rebalance training data across the 5 people if recognition favours Asher (231 images vs 61–121).
4. Expression detection to be re-enabled later; older analysis scripts still reference expression.
5. Merge/close PR #1 once the user is happy.

## Changelog
- 2026-10-06: Created this file.
- 2026-10-07: Learned `main` has substantial tested work (threaded UDP camera, two-pass detection, batched GPU inference, AMP training, uv packaging, Emberwatch app, 5-person retrain). Merged it into the PR branch with main as the base and rewrote this file to match. WebRTC app parked at `79ff481`.
- 2026-10-07: Recorded the user's preference to be walked through next steps step by step.
- 2026-10-07: Corrected the uv assumption (user uses pip). Added the honesty rule.
- 2026-10-07: User ran `main`'s app.py: starts fine (model loaded, 5 persons, cuda) but the video is still slow. Cause not yet known; asked for the app's on-screen fps and `live_inference.py`'s window-title FPS to tell compute-bound from browser-transport-bound.
