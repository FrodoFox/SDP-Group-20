import socket
import subprocess

with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Listen on all IPv6 interfaces
    s.bind(('::', 9000))
    s.listen(1)
    print("👂 Vision Pi (Freddy) is ONLINE and listening on port 9000...")
    
    while True:
        conn, addr = s.accept()
        with conn:
            if conn.recv(1024) == b"START":
                print(f"🚀 Signal received! Launching GolfBallTracker...")
                # Ensure this path is correct for your Pi
                subprocess.Popen(["python3", "/home/freddy/SDP-Group-20/vision_code/GolfBallDetection_picam2.py"])