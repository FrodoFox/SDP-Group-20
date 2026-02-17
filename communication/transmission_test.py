import xmlrpc.client
import socket

# --- CONFIG ---
RECEIVER_IP = "fe80::2ecf:67ff:fe0c:f9e5"
INTERFACE = "eth0"
PORT = 8000

TARGET_URL = f"http://[{RECEIVER_IP}%{INTERFACE}]:{PORT}/"

def test_transmitter():
    print(f"🚀 Initializing Transmitter...")
    print(f"🔗 Target: {TARGET_URL}")
    
    try:
        test_sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        test_sock.settimeout(2)
        test_sock.connect((RECEIVER_IP, PORT, 0, socket.if_nametoindex(INTERFACE)))
        test_sock.close()
        print("✅ Physical Link & Port: OPEN")
    except Exception as e:
        print(f"❌ Physical Link Failure: {e}")
        return

    try:
        proxy = xmlrpc.client.ServerProxy(TARGET_URL)
        print("📡 Sending test coordinates (10.5, 20.0, 30.2)...")
        
        response = proxy.send_coords(10.5, 20.0, 30.2)
        
        if response is True:
            print("✅ SUCCESS: weepinbell received the data!")
        else:
            print(f"⚠️ weepinbell responded with: {response}")
            
    except Exception as e:
        print(f"❌ RPC Function Call Failed: {e}")

if __name__ == "__main__":
    test_transmitter()