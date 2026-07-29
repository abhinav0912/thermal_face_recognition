"""
test_camera_connection.py
==========================
Run this ON YOUR MACHINE (same network as the FLIR A50) to figure out how
the camera can actually be reached before we wire up the live pipeline.

It does three things:
  1. Checks whether the camera answers on the common ports (80 = web UI,
     554 = RTSP, 3956 = GigE Vision discovery).
  2. Fetches the camera's web UI at http://<ip> so you can confirm it's
     alive and (more importantly) log in there yourself to find the exact
     RTSP path under its streaming/video settings.
  3. Tries a handful of RTSP URL patterns commonly used by FLIR / ONVIF
     network cameras. NOTE: these are educated guesses, not confirmed A50
     paths — treat this as a quick automated sweep, not a substitute for
     checking the web UI. If one connects, it saves a snapshot .jpg so you
     can visually confirm it's actually the thermal feed.

Usage:
    python test_camera_connection.py --ip 192.168.1.50
    python test_camera_connection.py --ip 192.168.1.50 --user admin --password secret
"""

import argparse
import socket
import urllib.request
import urllib.error

COMMON_PORTS = {80: "HTTP (web UI)", 554: "RTSP", 3956: "GigE Vision discovery"}

RTSP_PATH_CANDIDATES = [
    "",
    "live",
    "live.sdp",
    "live/ch0",
    "ch0",
    "ch1",
    "1",
    "stream1",
    "onvif-media/media.amp",
    "avc",
    "mpeg4",
]


def check_ports(ip: str):
    print(f"\n[1/3] Checking common ports on {ip} …")
    open_ports = []
    for port, desc in COMMON_PORTS.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((ip, port))
            status = "OPEN" if result == 0 else "closed"
            if result == 0:
                open_ports.append(port)
            print(f"  Port {port:>5} ({desc:<24}) : {status}")
    return open_ports


def check_web_ui(ip: str):
    print(f"\n[2/3] Fetching http://{ip}/ …")
    try:
        with urllib.request.urlopen(f"http://{ip}/", timeout=3) as resp:
            body = resp.read(2000).decode("utf-8", errors="ignore")
            print(f"  HTTP {resp.status} — page reachable.")
            if "<title>" in body.lower():
                start = body.lower().find("<title>") + 7
                end = body.lower().find("</title>", start)
                print(f"  Page title: {body[start:end].strip()!r}")
            print("  -> Log in here in a browser and look for a "
                  "'Streaming' / 'Video' / 'Network' settings page — "
                  "that's where the real RTSP URL (if any) is listed.")
            return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} — server responded (may need login). Still reachable.")
        return True
    except Exception as e:
        print(f"  Could not reach web UI: {e}")
        return False


def check_rtsp(ip: str, user: str, password: str):
    print(f"\n[3/3] Trying common RTSP URL patterns on {ip} (this can take a minute) …")
    try:
        import cv2
    except ImportError:
        print("  opencv-python not installed. Run: pip install opencv-python")
        return

    auth = f"{user}:{password}@" if user else ""
    found = None
    for path in RTSP_PATH_CANDIDATES:
        url = f"rtsp://{auth}{ip}:554/{path}"
        print(f"  Trying {url} …", end=" ", flush=True)
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            print("SUCCESS")
            found = url
            snapshot_path = "camera_snapshot.jpg"
            cv2.imwrite(snapshot_path, frame)
            print(f"  -> Saved a snapshot to {snapshot_path}. Open it and "
                  f"check it actually looks like the thermal feed.")
            break
        print("no")

    if found:
        print(f"\n  Working RTSP URL: {found}")
    else:
        print("\n  None of the guessed RTSP paths worked. This likely means "
              "either:\n"
              "    (a) the exact path is different — check the web UI's "
              "streaming settings, or\n"
              "    (b) the A50 doesn't do RTSP and needs GigE Vision "
              "(GenICam) instead — we'll use the 'harvesters' library for "
              "that path.")


def main():
    parser = argparse.ArgumentParser(description="Probe a FLIR A50 for how to stream from it")
    parser.add_argument("--ip", required=True, help="Camera IP address")
    parser.add_argument("--user", default="", help="RTSP username, if required")
    parser.add_argument("--password", default="", help="RTSP password, if required")
    args = parser.parse_args()

    open_ports = check_ports(args.ip)
    check_web_ui(args.ip)

    if 554 in open_ports or not open_ports:
        check_rtsp(args.ip, args.user, args.password)
    else:
        print("\n[3/3] Port 554 (RTSP) is not open — skipping RTSP probe.")
        if 3956 in open_ports:
            print("  Port 3956 is open, which suggests GigE Vision — "
                  "we'll use the GenICam/harvesters path instead.")


if __name__ == "__main__":
    main()
