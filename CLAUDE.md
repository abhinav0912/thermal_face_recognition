# CLAUDE.md: Thermal Face Recognition

Project memory for Claude Code. Read this first in every session, local or cloud.
Last rewritten 2026-11-02 for a **local** session on the user's PC (earlier versions were written for a cloud sandbox).

## Working rules (from the user)
- **Be truthful.** If you don't know something or haven't verified it, say "I do not know". Label unverified claims as unverified. Never present a guess as fact.
- **When asked for "next steps", walk the user through them one at a time** with exact PowerShell commands, and say what they should see after each step.
- The user is on **Windows 11, VS Code, PowerShell** (not Git Bash). Use `python -m ...` forms. Use backslash paths in commands.
- **This is a local session: you can run the real camera, the GPU and the app yourself.** Measure and verify things instead of asking the user to, except where a person has to stand in front of the camera. Earlier cloud sessions could not touch the camera or GPU, so anything marked "sandbox only" below has not been checked on the real hardware.
- Keep this file current. At the end of any turn that changes code, decisions, setup or open items, edit the relevant section and add a line to the Changelog.
- Before editing code: `git fetch origin`, check which branch you are on, and read `git log` (see "Git state"). Make small edits; don't overwrite tested code.

## What this project is
A research project at the Convergence Lab Initiative (CLI), Department of Computer Science, Virginia Commonwealth University. The user is a research assistant. Regular progress reports go to the research advisor.

Most face recognition depends on visible light and fails in the dark. This project does **face recognition from thermal (long-wave infrared) images**, which work in total darkness, with a model called **DualHeadFaceNet**: a MobileNetV2 backbone (RGB-pretrained), a shared 1280 to 512 layer, then two heads, **identity** and **expression**. The research idea is to do identity and expression together.

### Offline research results (July 2026 report, 113-subject dataset, 128x128 thermal images)
Identity accuracy 90-98%, expression accuracy 80-92%, five expression classes. Augmentation adds about 3.2%, joint training of both heads adds about 5%. Quantization cuts model size 36% and speeds inference 1.3-2x for 1.5% accuracy loss. Training was two-phase: first on paired RGB images (backbone frozen), then fine-tuned on thermal. Known limits: small dataset, sunglasses block thermal emission from the eyes, 128x128 resolution, no correction for ambient temperature, closed-set only.

## What the user is building now (the goal)
A **live demo system** on a FLIR A50 thermal camera:
- Show each person's **name** on their face box; show **Unknown** if not recognised.
- Recognise **several people in one frame**.
- **Collect face data** from the camera (video capture) to train on new people.
- **Enroll a new person instantly, without retraining** (gallery fallback), for live demos.
- A **browser interface** (Streamlit) styled as the chosen "Emberwatch" mockup. The page title must stay **"Thermal Face Recognition"** (Emberwatch was only a design codename).
- The live view should feel as smooth as the native OpenCV window.
- **Expression recognition is switched off on purpose for now** and will come back later. Search `expression disabled` in the code to find the pieces.

## Where things stand (2026-11-02)
- Live pipeline works on the real camera: RTSP stream, face detection, tracking, classifier plus gallery fallback, several faces per frame, instant enrollment, data collection tool, Streamlit app.
- Live model is **identity-only and trained on 5 people** (IDs 120-124), not the 113 subjects. The old 113-person set was retired from the working dataset. No held-out accuracy has been measured for the 5-person model.
- Speed is the open engineering problem: Streamlit Classic mode ran 4.0 fps, the new MJPEG "Fast stream" mode 11.5 fps, the native window 21.6 FPS (see Measured results).
- A progress report (original July content plus new live-system sections) exists; see "Progress report".

## Hardware and environment
- Camera: FLIR A50, link-local IP `169.254.0.82` over direct Ethernet. Stream: `rtsp://169.254.0.82:554/avc`. Camera web UI: `http://169.254.0.82`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120). Needs PyTorch built for CUDA 12.8. The `.venv` has torch `2.11.0+cu128`, `cuda.is_available()` True (verified 2026-10-07).
- Project folder: `C:\thermal_face_recognition-main\thermal_face_recognition-main` (double-nested, from a GitHub zip). Activate: `.\.venv\Scripts\Activate.ps1`.
- The venv has **no pip module** (`python -m pip` fails), which looks like a `uv`-created venv, but how it was made is not confirmed. The user says they use pip. `main` ships `pyproject.toml` and `uv.lock`. Everything the app needs was already installed. `requirements.txt` would pull CPU-only torch on a fresh venv; for a fresh install take torch from `https://download.pytorch.org/whl/cu128`.
- The network is behind a TLS-inspecting proxy (Zscaler); `pyproject.toml` sets `[tool.uv] system-certs = true` for that.
- Git is installed but needed `C:\Program Files\Git\cmd` added to PATH. `.streamlit` is a dot-folder.

## How to run (PowerShell, project folder, venv active)
```powershell
python live_inference.py                      # native window; n = enroll largest face, q = quit
python live_inference.py --source 0           # test with a webcam instead of the thermal camera
python -m streamlit run app.py                # browser app at http://localhost:8501
python collect_data.py                        # capture training data (preview, SPACE to start, q to quit)
python train.py --epochs 40 --batch_size 32   # full retrain from the pretrained backbone
python evaluate.py                            # held-out evaluation (reads data/ and checkpoints/)
```
Fast head-only retrain if time matters: `python train.py --epochs 8 --unfreeze_epoch 999 --batch_size 128`.
`train.py` scans `data/thermal-face-128x128/` and rebuilds `checkpoints/label_map.json` each run. Camera transport is UDP; override with env `THERMAL_CAMERA_RTSP_TRANSPORT=tcp` (not recommended, see gotchas).

## Measured results (user's PC)
| Date | Setup | Result |
|---|---|---|
| 2026-10-20 | `live_inference.py` native window, one person | **21.6 FPS** |
| 2026-10-20 | Streamlit Classic mode | **4.0 fps** |
| 2026-10-20 | Streamlit Fast stream, full-res detection, 2 tracked | **11.5 fps**: detect 64 ms, identify 13 ms, encode 1 ms per frame (about 78 ms) |
| 2026-11-02 | `detect()` only, one person at ~158 px face width, same 80 frames timed both ways | Full **52.1 ms**, Half **26.9 ms**; face found in 80/80 frames in both |
| 2026-11-02 | Same, one person close up (~305 px face width) | Full **45.2 ms**, Half **24.8 ms**; face found in 80/80 frames in both |

Detection (two Haar passes on the full frame) is about 80% of the frame time. Half-resolution detection is about 2x faster than full (detect stage only; the whole-pipeline fps with half resolution is **not yet measured**). Faces smaller than ~158 px have not been tried on the camera. Both 2026-11-02 runs had extra boxes in some frames (up to 3 full, 2 half) with only one person in view; likely false detections (only the last frame was inspected). How many people were in frame for the 21.6 figure is unknown. One older screenshot of the native window shows 27.5 FPS, so the native number varies.

## Architecture of the live system
```
FLIR A50 --RTSP/UDP--> ThermalCamera (reader thread, newest frame)
  -> ThermalFaceDetector (Haar + CLAHE, strict pass + relaxed pass, texture filter, dedup)
  -> MultiFaceTracker (one track per face, smoothing, jump rejection)
  -> classifier (identity head), batched on GPU; confidence >= 0.5 -> name
       else gallery match (cosine >= 0.6) -> name (live-match)
       else "Unknown"
  -> boxes drawn -> JPEG -> MJPEG stream -> browser <img>   (Streamlit draws only the UI around it)
```

## Code map
| File | Role |
|---|---|
| `model.py` | `DualHeadFaceNet`, `get_embedding()` (512-d), `forward_with_embedding()`. `expression_head` stays so old checkpoints load. |
| `train.py` | Identity-only training (expression loss commented out). Mixed precision, cudnn benchmark, `--workers` defaults to 0 on Windows (workers > 0 deadlocked). |
| `inference.py` | `FaceRecognizer`: `predict_image`, `predict_batch` (one GPU pass per frame), `embed_image`. |
| `camera.py` | `ThermalCamera`: background reader thread, `.read()` never blocks, `frame_id` counter, forces `cv2.CAP_FFMPEG`, UDP by default. |
| `face_detector.py` | `ThermalFaceDetector` (two-pass Haar, `detect_scale` 1.0 default, 0.5 = half-resolution), `FaceTracker` (one face), `MultiFaceTracker`. Crop has a bigger bottom margin (0.45) because the cascade cuts off the chin. |
| `gallery.py` | Instant enrollment: averaged embeddings in `data/gallery.json`. |
| `person_names.py` | id to name in `data/person_names.json`. |
| `live_inference.py` | Native-window live recognition. |
| `collect_data.py` | Video capture, about 4 frames/s into `{id}-TD-A-{n}.jpg` plus raw clip in `data/videos/`, and nine angle stills. Asks for a name on new IDs. |
| `app.py` | Streamlit UI (Emberwatch style). Settings: Run live feed toggle, **Video mode** (Fast stream = default, Classic = old per-frame loop kept as fallback), **Half-resolution detection** toggle (ON by default since 2026-11-02; switch it off for full-resolution detection), camera source, checkpoint dir, device, thresholds. Status bar shows fps, tracked count, frame size, per-stage ms. |
| `stream_server.py` | `RecognitionStreamer`: recognition thread plus MJPEG server (`/stream`, `/snapshot.jpg`, `/stats`) on a local port, 127.0.0.1 only unless env `THERMAL_STREAM_HOST` is set. No authentication. |
| `.streamlit/config.toml` | Dark and amber theme. |
| `test_camera_connection.py` | One-off probe for camera ports and RTSP paths. |
| Older scripts | `benchmark.py`, `ablation_study.py`, `error_analysis.py`, `visualization*.py`, `model_export.py` still reference expression and were deliberately left alone. `complete_pipeline.py`, `prepare_data.py` and `extract_metrics.py` also exist and were not reviewed. Many `*.md` files in the repo root are older generated docs and may be stale. |

## Data and people
`data/person_names.json`: 120 Abhinav (the user), 121 Chandhini, 122 Jyothsna, 123 Lakshmi, 124 Asher. Images in `data/thermal-face-128x128/`: 120 has 121, 121 has 61, 122 has 113, 123 has 62, 124 has 231. Uneven counts may bias the classifier toward 124 (unchecked). Delete bad captured frames by filename; numbering gaps are harmless. Checkpoints: `checkpoints/` (current, 5 people), `checkpoints_backup_4people_epoch7/` (previous). The old 113-person data was moved out of the working set (`data/Temp/` is gitignored).

## Decisions and gotchas (don't re-learn these)
- `opencv-python` is pinned `<5.0.0` (5.0 had no `CascadeClassifier`).
- **RTSP: TCP times out opening the stream on this network (30 s); UDP connects at once.** UDP prints "corrupted macroblock" warnings; they are harmless. Don't force TCP.
- Haar on thermal frames gives false positives (hairline, chest, dark background) and overlapping boxes, so detection is two-pass with dedup and trackers reject jumps. The classifier is closed-set, so it can name a non-face confidently. **Open bug:** with 2 tracked, a second box on the dark background (probably a false detection, unverified) was labelled with a real name at 95%.
- Gallery matching is weaker than a trained identity. Anyone worth keeping should be retrained into the classifier.
- Streamlit rules: no blank line inside an injected `<style>` block (the markdown parser prints the CSS as text). Don't loop `st.rerun()` (flicker). Toggle uses `key=`, not `value=<own state>` (first click got dropped). Streamlit's CSS sets images to `object-fit: scale-down`; the stream `<img>` overrides it inline. Streamlit fades elements while a rerun runs, which probably explains dim labels in Classic mode (unverified).
- The detector is **not thread-safe**. `RecognitionStreamer.set_active(False)` waits for the in-flight frame so enrollment can use the detector safely.
- **Unproven diagnosis:** in a cloud-sandbox test with fake camera and model, Classic mode ran about 18-22 fps, so "Streamlit per-frame overhead" may not fully explain the 4.0 fps. The cause on the real PC is not established; the per-stage timings in the status bar exist to find it.
- **Half-resolution detection, sandbox benchmark only** (real dataset face crops pasted on a noisy dark background, not representative): 3.4x to 14x faster, similar detection rate for faces of about 220 px or more, clearly fewer detections for faces of about 100 px (10 of 40 against 25 of 40 at 640x480). Real effect on the demo is unknown until tried on the camera.
- WebRTC (`streamlit-webrtc` with a custom server-side video track) was built once and never run; it is in git history at commit `79ff481`. Revive only if the MJPEG stream is smooth server-side but looks choppy in the browser.

## Roadmap
**Next (small, concrete)**
1. ~~Time Half-resolution detection~~ Done for `detect()` (see Measured results); half is now the app default with an on/off toggle. Still open: whole-pipeline fps in the app, and faces smaller than ~158 px.
2. Fix the false box that gets a real name: stricter face test, or require a stable track before naming, or an "is this a face" check.
3. Test with two real people in frame (the two-pass detection and dedup were built for this).
4. Rebalance images across the 5 people, retrain, and **measure held-out accuracy**, including unknown-person rejection with people the model has never seen.
5. Merge PR #1 so `main` has the fast stream and the other changes.

**If speed is still short of the native 21.6 FPS:** run detection and identification in parallel threads, or detect on every second frame (lags when people move).

**Later**
- Re-enable expression recognition in the live pipeline (the head and checkpoints still support it).
- Video-based temporal training (use motion across frames instead of single images), the long-term research direction from the July report.
- Address the report's open limits: more subjects, sunglasses/eyewear, ambient-temperature correction, 128x128 resolution.

## Progress report
- The July 2026 report (5 pages, written in Google Docs) was extended with new sections on the live system: Claude Doc https://claude.ai/artifact/JoiCYoCEHEBDZGVrEdZXEB . The user then edited their own Google Doc copy (renumbered sections, "Progress Report (October)").
- A corrected version of that Google Doc (section references, formula, figure captions, date, dashes, hyphens, grammar) was produced as a Word file, but it was **not** applied to the user's Google Doc (no Google Docs editing tool was available). The user was given steps to paste it in.
- Still the user's call in the report: the phrase "lab PC" was assumed, "tested in a headless browser here" in 5.8, a table note "number of people not recorded", the Executive Summary does not mention the live system, Figure 5 shows 27.5 FPS while the text says 21.6 FPS, Figure 6 shows other people's names.
- Update the report's numbers if Half-resolution timings or held-out accuracy are measured.

## Git state
- Repo: `abhinav0912/thermal_face_recognition`. Branches: `main` (tested earlier on the real camera, still lacks the fast stream, Half-resolution option and toggle fix) and `claude/thermal-face-recognition-access-hyfolm` (adds `stream_server.py`, Video mode and Face detection selectors, `ThermalCamera.frame_id`, toggle fix, this file). **PR #1** is open for that branch; pushing to the branch updates it. Don't open another PR.
- As of 2026-11-02 `origin/main` is `f655745`; the branch tip was `3530bad`. Check with `git fetch origin` and `git log --oneline origin/main -5`.
- The user's local checkout was switched to the branch on 2026-10-20 (their `git status` showed it). Confirm with `git branch --show-current`.
- Windows renames re-downloaded files to `name (1).py`. Update code through git, not by downloading files.
- Commits may be authored by this local session or by earlier cloud sessions. Don't force-push or rewrite history. Optional commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>`.

## Design (Emberwatch)
Background `#16130f`, surface `#211c17`, thermal amber accent `#ff7a2f`, good `#5fd98a`, info `#4fc3e8`, alert `#ff5c72`. Fonts: Archivo Expanded (headings), IBM Plex Sans (body), IBM Plex Mono (data). Chosen mockup: https://claude.ai/artifact/Sv5fAR9WoP9gKk26dGSJoi (five unchosen alternatives: https://claude.ai/artifact/Kn7TvE2UA5dsD1AD5umHML).

## Changelog
- 2026-10: Camera link found and working over RTSP/UDP. Data collection, tracking and live recognition built. Gallery fallback for instant enrollment. Streamlit app in Emberwatch style. Expression disabled on purpose. Retrained on 5 people (IDs 120-124).
- 2026-10-20: User measured 21.6 FPS (native), 4.0 fps (Classic), 11.5 fps (Fast stream). Built MJPEG Fast stream (`stream_server.py`) and the Half-resolution option; fixed the two-click toggle bug. Sandbox-tested only.
- 2026-10-30: Progress report extended with live-system sections; user edited it; corrected Word version delivered (not applied to the user's Google Doc).
- 2026-11-02: CLAUDE.md rewritten for a local session: what the project is, goals, current state, how to run, roadmap, report status.
- 2026-11-02 (local session): Timed full vs half detection on the real camera (see Measured results). `app.py`: the "Face detection" selectbox became a "Half-resolution detection" toggle, ON by default. Checked with Streamlit AppTest that the toggle sets `detector.detect_scale` to 0.5/1.0; not yet looked at in the browser. `live_inference.py` (native window) and the `ThermalFaceDetector` default are unchanged (full). Note: the venv here is `venv\`, not `.venv\`.
