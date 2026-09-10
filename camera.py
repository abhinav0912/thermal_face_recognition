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

# Must be set before any cv2.VideoCapture(...) call that opens an RTSP URL.
# OpenCV's FFmpeg backend defaults to UDP for RTSP, which silently drops lost
# packets and corrupts frames mid-decode (visible as "corrupted macroblock" /
# "error while decoding MB" spam from FFmpeg). TCP retransmits instead, which
# fixes that at the cost of slightly higher latency on a lossy link.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2

DEFAULT_RTSP_URL = "rtsp://169.254.0.82:554/avc"


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
            # Force the FFmpeg backend explicitly for URL sources (RTSP etc).
            # Without this, OpenCV auto-picks a backend, and on some
            # opencv-python builds/platforms that pick isn't guaranteed to be
            # FFmpeg -- in which case OPENCV_FFMPEG_CAPTURE_OPTIONS above
            # (rtsp_transport;tcp) silently does nothing, since it's read by
            # the FFmpeg backend specifically.
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
