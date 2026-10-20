"""
stream_server.py
================
Fast live-video path for app.py: runs the recognition loop in a background
thread and serves the annotated frames as an MJPEG stream (HTTP
multipart/x-mixed-replace) that the browser plays natively in a plain <img>.

Why: with the st.fragment + st.image approach, every video frame is a full
Streamlit rerun plus an image upload to the browser. Measured on the demo PC
that capped the page at ~4 fps, while live_inference.py (same camera, detector,
model) runs at ~21 fps natively -- so the cost was Streamlit's per-frame
round-trip, not the recognition. Here Streamlit never touches individual
frames: this module detects/tracks/identifies on its own thread (exactly the
same steps as live_inference.py) and pushes JPEGs straight to the browser.
Streamlit only draws the surrounding UI.

Standard library + OpenCV only (no Streamlit import), so it can be tested on
its own. app.py supplies the identify/annotate callables, so this module
doesn't duplicate any recognition logic.

Security: the stream has no authentication, so it binds to 127.0.0.1 by
default (only the PC itself can open it). Set THERMAL_STREAM_HOST=0.0.0.0 to
let other machines open the page's video; anyone who can reach that port
can then watch the feed.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

STREAM_HOST = os.environ.get("THERMAL_STREAM_HOST", "127.0.0.1")
STREAM_PORT = int(os.environ.get("THERMAL_STREAM_PORT", "0"))  # 0 = pick a free port
BOUNDARY = "frame"


class RecognitionStreamer:
    """
    identify_fn(recognizer, gallery_data, crops_bgr, unknown_threshold,
                gallery_threshold) -> [(label, known), ...]   (one per crop)
    annotate_fn(frame, box, label, known) -> None   (draws on frame in place)
    """

    def __init__(self, detector, tracker, identify_fn, annotate_fn,
                 gallery_loader, gallery_path, jpeg_quality: int = 80):
        self.detector = detector
        self.tracker = tracker
        self.identify_fn = identify_fn
        self.annotate_fn = annotate_fn
        self.gallery_loader = gallery_loader
        self.gallery_path = gallery_path
        self.jpeg_quality = jpeg_quality

        self._cfg = None            # (camera, recognizer, unknown_thr, gallery_thr)
        self._cfg_lock = threading.Lock()
        self._active = threading.Event()
        self._work_lock = threading.Lock()   # held while a frame is being processed
        self.generation = 0         # bumps each time the stream is switched on

        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0

        self._stats = {"fps": 0.0, "tracked": 0, "error": "", "ms": {}}
        self._ms = {}               # smoothed per-stage timings (ms), to show where time goes
        self.frames_sent = 0        # JPEGs written to browsers, across all connections
        self._gallery = {}
        self._gallery_mtime = None

        self._server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), self._make_handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- control (called from the Streamlit script thread) ----

    def configure(self, camera, recognizer, unknown_threshold, gallery_threshold):
        with self._cfg_lock:
            self._cfg = (camera, recognizer, unknown_threshold, gallery_threshold)

    def set_active(self, active: bool):
        if active:
            if not self._active.is_set():
                self.generation += 1
                self._active.set()
        else:
            self._active.clear()
            # Wait for any frame already in flight, so the caller can safely use the
            # (not thread-safe) detector itself right after pausing -- e.g. enrollment.
            with self._work_lock:
                pass

    @property
    def active(self) -> bool:
        return self._active.is_set()

    def stats(self) -> dict:
        return dict(self._stats)

    def url(self, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}/stream?g={self.generation}"

    # ---- processing thread ----

    def _refresh_gallery(self):
        try:
            mtime = os.path.getmtime(self.gallery_path)
        except OSError:
            mtime = None
        if mtime != self._gallery_mtime:
            self._gallery = self.gallery_loader()
            self._gallery_mtime = mtime

    def _loop(self):
        last_id = None
        times = []
        while True:
            self._active.wait()
            with self._work_lock:
                if not self._active.is_set():
                    continue
                with self._cfg_lock:
                    cfg = self._cfg
                if cfg is None:
                    time.sleep(0.05)
                    continue
                camera, recognizer, unknown_thr, gallery_thr = cfg

                fid = getattr(camera, "frame_id", None)
                if fid is not None and fid == last_id:
                    time.sleep(0.002)   # no new camera frame yet
                    continue
                ok, frame = camera.read()
                if not ok or frame is None:
                    self._stats["error"] = "No frame received"
                    time.sleep(0.05)
                    continue
                last_id = fid

                t0 = time.perf_counter()
                try:
                    self._refresh_gallery()
                    tracks = self.tracker.update(self.detector.detect(frame))
                    t1 = time.perf_counter()
                    boxes, crops = [], []
                    for box in tracks.values():
                        crop = self.detector.crop(frame, box)
                        if crop.size == 0:
                            continue
                        boxes.append(box)
                        crops.append(crop)
                    labels = self.identify_fn(recognizer, self._gallery, crops,
                                              unknown_thr, gallery_thr) if crops else []
                    t2 = time.perf_counter()
                    for box, (label, known) in zip(boxes, labels):
                        self.annotate_fn(frame, box, label, known)
                except Exception as e:   # keep the stream alive on a bad frame
                    self._stats["error"] = f"{type(e).__name__}: {e}"
                    time.sleep(0.05)
                    continue

                now = time.time()
                times.append(now)
                times = times[-30:]
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0

                cv2.putText(frame, f"{fps:.1f} FPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                t3 = time.perf_counter()
                for k, v in (("detect", t1 - t0), ("identify", t2 - t1), ("encode", t3 - t2)):
                    self._ms[k] = v * 1000 if k not in self._ms else 0.9 * self._ms[k] + 0.1 * v * 1000
                self._stats = {"fps": fps, "tracked": len(boxes), "error": "", "ms": dict(self._ms)}
                if ok:
                    with self._cond:
                        self._jpeg = buf.tobytes()
                        self._seq += 1
                        self._cond.notify_all()

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    # ---- HTTP ----

    def _make_handler(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/stream":
                    self._stream()
                elif path == "/stats":
                    body = json.dumps({**streamer.stats(), "active": streamer.active,
                                       "frames_sent": streamer.frames_sent}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/snapshot.jpg":
                    jpeg, _ = streamer.wait_for_frame(-1, 0)
                    if jpeg is None:
                        self.send_error(404, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(jpeg)
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                last = -1
                try:
                    while True:
                        jpeg, seq = streamer.wait_for_frame(last, 1.0)
                        if seq == last or jpeg is None:
                            if not streamer.active:
                                break        # feed paused: free this connection
                            continue
                        last = seq
                        self.wfile.write(
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        streamer.frames_sent += 1
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass

        return Handler


def embed_html(url: str) -> str:
    """The <img> that plays the stream. Rendered once by Streamlit; the browser
    then pulls frames itself with no further Streamlit involvement. (object-fit is
    set explicitly because Streamlit's own CSS makes images `scale-down`, which would
    leave a small frame un-stretched inside a black box.)"""
    return (f'<img src="{url}" alt="Live thermal feed" '
            f'style="width:100%;max-width:1024px;object-fit:contain;border-radius:4px;'
            f'display:block;background:#000;">')
