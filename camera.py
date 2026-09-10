"""
camera.py
=========
Frame source for the live thermal pipeline.

The FLIR A50 at 169.254.0.82 was confirmed reachable over RTSP at
rtsp://169.254.0.82:554/avc (see test_camera_connection.py). This wraps
cv2.VideoCapture with automatic reconnect, since RTSP streams from network
cameras can drop or stall.

--source also accepts a plain webcam index (e.g. "0") for testing the
pipeline without the thermal camera attached.
"""

import os
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
# Override with the THERMAL_CAMERA_RTSP_TRANSPORT env var ("tcp" or "udp")
# to A/B test which one actually behaves better on your link.
RTSP_TRANSPORT = os.environ.get("THERMAL_CAMERA_RTSP_TRANSPORT", "tcp")


class ThermalCamera:
    def __init__(self, source=DEFAULT_RTSP_URL, reconnect_delay: float = 2.0):
        self.source = _parse_source(source)
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._open()

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

    def read(self):
        """Return (ok, frame_bgr). Attempts one reconnect on failure."""
        ok, frame = self.cap.read()
        if not ok:
            print(f"  [camera] Lost connection, reconnecting to {self.source} ...")
            time.sleep(self.reconnect_delay)
            try:
                self._open()
                ok, frame = self.cap.read()
            except ConnectionError:
                ok, frame = False, None
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


def _parse_source(source):
    """Allow '0', '1', etc. (webcam index) to come through as plain strings from argparse."""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source
