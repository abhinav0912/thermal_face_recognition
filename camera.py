"""
camera.py
=========
Frame source for the live thermal pipeline.

The FLIR A50 at 169.254.0.82 was confirmed reachable over RTSP at
rtsp://169.254.0.82:554/avc (see test_camera_connection.py). This wraps
cv2.VideoCapture with automatic reconnect, since RTSP streams from network
cameras can drop or stall.

Frames are read continuously in a background thread rather than on the
calling thread. A network read (or, worse, a post-drop reconnect that
redoes the whole RTSP handshake) can take anywhere from milliseconds to
several seconds -- fine for a CLI loop, but fatal for Streamlit, where
every rerun's `camera.read()` call blocks that entire rerun. On the old
synchronous design, one slow read froze the whole UI for its duration.
Now `.read()` just returns whatever the background thread most recently
decoded, so it's effectively instant, and a stalled reconnect only slows
down how fresh that cached frame is -- it doesn't block anyone.

--source also accepts a plain webcam index (e.g. "0") for testing the
pipeline without the thermal camera attached.
"""

import os
import threading
import time

import cv2

DEFAULT_RTSP_URL = "rtsp://169.254.0.82:554/avc"

# Which RTSP transport OpenCV's FFmpeg backend uses. UDP just drops lost
# packets (corrupted/glitchy frames, "corrupted macroblock" spam, but the
# stream keeps moving). TCP retransmits lost packets instead -- fine on a
# healthy link, but if the actual problem is a flaky physical connection
# (not just "UDP being UDP"), TCP means every drop now stalls the stream
# waiting on a retry instead of just glitching one frame, which can look
# like the feed hanging entirely rather than occasional bad frames.
# Confirmed on this A50/network: TCP times out opening the stream at all
# (30s FFmpeg timeout -> ConnectionError); UDP connects immediately. Default
# to UDP; override with THERMAL_CAMERA_RTSP_TRANSPORT=tcp if a future link
# behaves the other way around.
RTSP_TRANSPORT = os.environ.get("THERMAL_CAMERA_RTSP_TRANSPORT", "udp")


class ThermalCamera:
    def __init__(self, source=DEFAULT_RTSP_URL, reconnect_delay: float = 2.0):
        self.source = _parse_source(source)
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._stop = threading.Event()

        self._open()  # synchronous on construction, same as before: fail fast if the source is bad
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        if isinstance(self.source, str):
            # Must be set before cv2.VideoCapture(...) opens the URL, and the
            # FFmpeg backend must be explicit -- without an apiPreference,
            # OpenCV auto-picks a backend, and on some opencv-python
            # builds/platforms that pick isn't guaranteed to be FFmpeg, which
            # would silently no-op this option entirely.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            # A webcam index needs the platform's normal camera backend, not FFmpeg.
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Could not open camera source: {self.source}")

    def _reader_loop(self):
        """Background thread: keep the latest frame ready so .read() never blocks on I/O."""
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                print(f"  [camera] Lost connection, reconnecting to {self.source} ...")
                time.sleep(self.reconnect_delay)
                try:
                    self._open()
                except ConnectionError:
                    pass  # will just retry next loop iteration
                continue
            with self._frame_lock:
                self._latest_frame = frame

    def read(self):
        """Return (ok, frame_bgr) -- the most recently decoded frame, without blocking."""
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


def _parse_source(source):
    """Allow '0', '1', etc. (webcam index) to come through as plain strings from argparse."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source
