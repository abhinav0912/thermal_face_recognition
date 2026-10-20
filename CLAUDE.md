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
- The Claude sandbox cannot reach the camera, has no GPU, and cannot install torch (the PyTorch download host is blocked; PyPI works). It DOES have headless Chromium (`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`) and can pip-install streamlit/opencv-headless/playwright into a scratch venv. So UI behaviour and the stream transport can be tested here with a **stub model + fake camera** (test harness lives in the session scratchpad, not in the repo), but real-camera speed, GPU and recognition quality can only be verified by the user running things locally. Ask for screenshots/terminal output.
- **Delivering files to the user:** committing to the branch does not give them a download. Use `SendUserFile` for files. Windows renames a re-downloaded file `name (1).py` when `name.py` already exists in the folder (this happened: `app (1).py`, `app (2).py`, `camera (1).py` in their project folder). Prefer making them update via git (`git pull origin claude/thermal-face-recognition-access-hyfolm`) instead of downloading. They were walked through: `git status` → `git restore app.py camera.py` → delete the downloaded `stream_server.py` → `git fetch origin` → `git checkout claude/thermal-face-recognition-access-hyfolm`. Whether they completed that is not confirmed.

## Code map (state of `main` at `f655745`)
| File | Role |
|---|---|
| `model.py` | `DualHeadFaceNet` (MobileNetV2, `identity_head`, `expression_head`), `get_embedding()`, `forward_with_embedding()` (one backbone pass gives logits and the gallery embedding). `expression_head` stays so old checkpoints load. |
| `train.py` | Identity-only training (expression loss commented out). Scans `data/thermal-face-128x128/`, rebuilds `checkpoints/label_map.json`. `cudnn.benchmark`, mixed precision (autocast + GradScaler), OS-aware `--workers` (0 on Windows: workers > 0 deadlocked). Full retrain from pretrained backbone each run. |
| `inference.py` | `FaceRecognizer`: `predict_image`, `predict_batch` (one batched GPU pass per frame), `embed_image`, name lookup. |
| `camera.py` | `ThermalCamera`: **background reader thread** (plus a `frame_id` counter so consumers can skip already-seen frames), so `.read()` returns the latest frame instantly and never blocks. `cv2.CAP_FFMPEG` for URLs. RTSP transport defaults to **UDP** (env `THERMAL_CAMERA_RTSP_TRANSPORT=tcp` to switch). |
| `face_detector.py` | `ThermalFaceDetector`: Haar + CLAHE, **two passes** (strict, then relaxed fallback for off-angle faces) with a grey-value std texture filter on relaxed boxes, overlap-coefficient dedup, per-pass cascade instances and try/except around `detectMultiScale` (intermittent OpenCV crash). `detect_scale` (default 1.0 = unchanged behaviour; 0.5 runs both Haar passes on a downscaled CLAHE frame and maps boxes back; min sizes scaled, floor 24). Crop uses a bigger bottom margin (cascade cuts the chin). `FaceTracker` (one face) and `MultiFaceTracker` (many faces, proximity matching, smoothing, outlier rejection, recovery). |
| `gallery.py` | Instant enrollment: averaged embeddings in `data/gallery.json`, cosine match (default 0.6). |
| `person_names.py` | id → name, `data/person_names.json`. |
| `live_inference.py` | OpenCV-window live recognition: classifier first, gallery fallback below 0.5 confidence, else Unknown. `n` enrolls the largest face, `q` quits. |
| `collect_data.py` | Video capture (preview, SPACE to start, ~4 frames/s into `{id}-TD-A-{n}.jpg`, raw clip in `data/videos/`) and angle stills. Asks for a name on new IDs. |
| `app.py` | Streamlit UI in Emberwatch style: two-column layout, no native sidebar. Settings panel: `Run live feed` toggle (**`key="running"`**; the pause button uses `on_click`), **Video mode** (`Fast stream (recommended)` = MJPEG via `stream_server.py`; `Classic (slower)` = old `st.fragment(run_every=0.05)` + `st.image` loop, kept as fallback), **Face detection** (`Full resolution (current)` default / `Half resolution (faster)` → sets `detector.detect_scale` on the shared detector, applies to stream, classic and enrollment), camera source, checkpoint dir, device, thresholds. Status bar (stream mode: fragment refreshed every 1 s) shows LIVE/IDLE, fps, tracked count, frame size, per-stage ms (`detect · identify · encode`), GPU. Camera, tracker, detector, recognizer and the streamer are `st.cache_resource` (one shared instance for all tabs). Switching stream→classic deactivates the streamer first (shared, not thread-safe detector). |
| `stream_server.py` | **Fast video path (added on the PR branch, untested on real hardware).** `RecognitionStreamer`: background thread runs detect → track → identify → annotate → JPEG, and serves it as an MJPEG stream (`/stream`, also `/snapshot.jpg`, `/stats`) on a local port (127.0.0.1 by default; env `THERMAL_STREAM_HOST`/`THERMAL_STREAM_PORT`). `app.py` embeds it as a plain `<img>` (inline `object-fit:contain`, because Streamlit's CSS makes images `scale-down`), so Streamlit never handles individual frames. Stream URL carries `?g=<generation>` so a pause/resume remounts the image. `/stats` returns fps, tracked, per-stage ms, frame size, `frames_sent`. `set_active(False)` blocks until any in-flight frame finishes, so enrollment can safely use the detector right after pausing. Gallery is reloaded only when `data/gallery.json`'s mtime changes. No auth: binds 127.0.0.1 unless `THERMAL_STREAM_HOST` is set. |
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
- **Unproven diagnosis:** in the sandbox, with a stub detector/model and a fake camera, the *classic* Streamlit path ran ~18–22 fps, so "Streamlit per-frame overhead" alone may not explain the user's 4 fps. The real cause on their PC is not established; the stream's per-stage timings are there to find it.
- **Second false box (seen 2026-10-20):** with 2 tracked, the right-hand box sat on the dark background (looks like a false detection, not verified) and was labelled "Chandhini (95%)". Closed-set classifier says a known name even for non-faces. Not fixed. A stricter face/texture filter or a face-vs-non-face check would be the fix.
- Streamlit fades elements while a rerun is in progress; the faint/dim label text seen in the classic mode is probably that (unverified). The stream mode renders at full brightness.
- Streamlit's own CSS sets images to `object-fit: scale-down`; override inline if a small frame must stretch.
- Streamlit toggle: do not pass `value=<its own previous state>` (identity flips, first click dropped). Use `key=`.
- Half-resolution detection, sandbox benchmark only (synthetic frames: real dataset face crops pasted on a noisy dark background, 4-core CPU, NOT representative of real thermal frames or the user's PC): detect time 3.4x–7x faster at 640x480 and 10x–14x at 1280x960; detection rate was similar for faces >= ~220 px but clearly worse for ~100 px faces (10/40 vs 25/40 at 640x480) and boxes agreed with full-res in only about half of the hits. So Half trades small/distant faces for speed. Real-world effect on the demo is unknown until the user tries it.
- Streamlit's browser path is inherently slower than `live_inference.py`'s native window; the user wants native-like smoothness. Whether `main`'s current `app.py` is fast enough is **not yet confirmed**.
- **WebRTC option (parked):** an `streamlit-webrtc` version of `app.py` with a custom server-side `VideoStreamTrack` exists in git history at commit `79ff481` (old branch tip). It was never run. Revive it only if `main`'s Streamlit feed is still too laggy. It predates the camera reader thread and the UDP default, so it needs re-basing on those.
- Training data: the old 113-person set was deleted (it caused heavy class imbalance). The roster is whoever has been collected live.
- Delete bad captured frames by filename; numbering gaps are harmless.

## Measured results (user's PC, FLIR A50, RTX PRO 6000)
| Date | Setup | Result |
|---|---|---|
| 2026-10-20 | `live_inference.py` native window, one person | **21.6 FPS** |
| 2026-10-20 | Streamlit Classic path (`main`'s app.py) | **4.0 fps** |
| 2026-10-20 | Streamlit **Fast stream**, Full-res detection, 2 tracked | **11.5 fps** (video overlay 10.9); detect 64 ms, identify 13 ms, encode 1 ms per frame (~78 ms total) |
Detection (two Haar passes on the full frame) is ~80% of the frame time now. Half-res detection and the Classic fallback have not been run by the user yet. How many people were in frame for the 21.6 figure is unknown.

## People (all 5 are in the trained classifier)
`data/person_names.json`: 120 Abhinav, 121 Chandhini, 122 Jyothsna, 123 Lakshmi, 124 Asher. Images in `data/thermal-face-128x128/` per id: 120 → 121, 121 → 61, 122 → 113, 123 → 62, 124 → 231. Uneven counts can bias the classifier toward 124.

## Git state
- Repo: `abhinav0912/thermal_face_recognition`. My branch: `claude/thermal-face-recognition-access-hyfolm`. **PR #1** already exists (don't create another; pushing updates it).
- 2026-10-07: the user chose `main` as the base. I merged `origin/main` (`f655745`) into my branch and took main's version of every file, so the branch tree equals `main` plus this updated `CLAUDE.md`. My earlier work stays in history.
- 2026-10-20 commits on the branch: `7786cb7` (fast stream + toggle fix) plus the later detection-resolution commit. The user's local checkout was on `main` and had downloaded copies; they need to switch to the branch (see Environment) or merge PR #1. `main` itself still lacks all of this until PR #1 is merged.
- Day to day, commits come from both sessions. Fetch before changing anything; prefer small, additive edits to files `main` has changed.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01TJZd3mBmQxiwUcHN8fftky`.

## Design (Emberwatch)
Dark warm charcoal `#16130f`, surface `#211c17`, thermal amber accent `#ff7a2f`, good `#5fd98a`, info `#4fc3e8`, alert `#ff5c72`. Fonts: Archivo Expanded (headings), IBM Plex Sans (body), IBM Plex Mono (data). Mockups: https://claude.ai/artifact/Sv5fAR9WoP9gKk26dGSJoi (chosen) and https://claude.ai/artifact/Kn7TvE2UA5dsD1AD5umHML (5 alternatives, not chosen).

## Open items
1. **User tests Half-resolution detection** (Settings → Face detection) with the fast stream: report fps and the `detect · identify · encode` ms, and whether people/faces are missed at their normal distance. Full resolution stays the default and the fallback.
2. User should confirm they are running the branch code (stream mode visible in Settings) via git, not via downloaded `(1)` copies. Their screenshot at 7:11 PM showed Fast stream running, so the files did get in place.
3. Second false box labelled with a real name (see gotchas). Candidate fixes: stricter filter, require tracker persistence before naming, add an "is a face" check. Not started.
4. If still slower than the native 21.6 FPS after Half-res: pipeline detection and identification on separate threads (cap = slowest stage), or detect every 2nd frame (lags when moving). Not built.
5. Live test with two real people in frame (two-pass detection and dedup were added for this).
6. Rebalance training data across the 5 people if recognition favours Asher (231 images vs 61–121).
7. Expression detection to be re-enabled later; older analysis scripts still reference expression.
8. Merge/close PR #1 once the user is happy; then they can stay on `main` and `git pull`.
9. Unanswered question to the user: is the person in the 2026-10-20 screenshots really Chandhini (name shown 80–95%)? If not, that is a misrecognition to investigate.
10. WebRTC (commit `79ff481`) stays parked; only revisit if the MJPEG stream is smooth on the server side but looks choppy in the browser.

## Changelog
- 2026-10-06: Created this file.
- 2026-10-07: Learned `main` has substantial tested work (threaded UDP camera, two-pass detection, batched GPU inference, AMP training, uv packaging, Emberwatch app, 5-person retrain). Merged it into the PR branch with main as the base and rewrote this file to match. WebRTC app parked at `79ff481`.
- 2026-10-07: Recorded the user's preference to be walked through next steps step by step.
- 2026-10-07: Corrected the uv assumption (user uses pip). Added the honesty rule.
- 2026-10-07: User ran `main`'s app.py: starts fine (model loaded, 5 persons, cuda) but the video is still slow. Cause not yet known; asked for the app's on-screen fps and `live_inference.py`'s window-title FPS to tell compute-bound from browser-transport-bound.
- 2026-10-20: Measured by the user on the same machine, one person in frame: `live_inference.py` native window **21.6 FPS**; Streamlit `app.py` status bar **4.0 fps** (page claims a 20fps target). Same detector/model/camera in both, so the model is not the bottleneck; the loss is in the Streamlit per-tick rerun / image transport (exact split inside that is NOT profiled). Next: replace the per-frame `st.image` path (candidates: MJPEG stream from a background thread, or the parked WebRTC version); neither is tested. Also noted: box label text is dark green on a dark frame and hard to read.
- 2026-10-20: Built the MJPEG fast-stream mode (`stream_server.py`, `Video mode` selector in `app.py`, `ThermalCamera.frame_id`). Also fixed a bug reproduced on main's original `app.py` (sandbox Streamlit 1.66): the `Run live feed` toggle needed two clicks to turn off because `value=` changed with its own state; now `key="running"` and the pause button uses `on_click`. Sandbox-tested only with stubs.
- 2026-10-20: User ran the fast stream on the real camera: **11.5 fps** (was 4.0 in Classic; native window 21.6). Per-frame: detect 64 ms, identify 13 ms, encode 1 ms. Observed a second, probably false box on the background labelled "Chandhini (95%)".
- 2026-10-20: Added `detect_scale` to `ThermalFaceDetector` (default 1.0 = identical to before) and a **Face detection** selector in `app.py` (Full = default/fallback, Half = faster), plus frame size in the status bar. Sandbox: switching Full↔Half live works with no errors; with the real Haar detector on synthetic frames detect time dropped ~120–130 ms → ~25 ms (not representative of the user's PC). Detection-rate trade-off for small faces recorded in gotchas.
- 2026-10-20: Delivery notes: files must be sent with `SendUserFile` (commits are not downloads); duplicates `name (1).py` come from re-downloading into a folder that already has the file; use git to update.
