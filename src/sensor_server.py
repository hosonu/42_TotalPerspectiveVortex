from math import e
from operator import truth
import socket
import pickle
import struct
import time
import argparse

from mybci import build_epochs, epochs_to_Xy

def start_server(host, port, subject, runs):
    print(f"Loading EEG data for Subject {subject}, Runs {runs}...")

    epochs = build_epochs(subject, runs=runs)
    X, y = epochs_to_Xy(epochs)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"\n[Hardware Mock] Listening on {host}:{port}...")
    print("Waiting for AI Client to connect...")

    try:
        conn, addr = server_socket.accept()
        print(f"\n[Hardware Mock] Connected by AI Client: {addr}")
        print("[Hardware Mock] Starting EEG data stream...\n")

        for i in range(len(X)):
            trial_data = X[i:i+1]
            truth_label = y[i]

            payload = (trial_data, truth_label)
            data_bytes = pickle.dumps(payload)

            conn.sendall(struct.pack('>I', len(data_bytes)))
            conn.sendall(data_bytes)

            print(f"Sent Trial {i+1}/{len(X)} (Size: {len(data_bytes)} bytes)")

            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n[Hardware Mock] Server stopped by user (KeyboardInterrupt).")
    except ConnectionResetError:
        print("\n[Hardware Mock] Client disconnected.")
    finally:
        try:
            conn.sendall(struct.pack('>I', 0))
        except:
            pass
        conn.close()
        server_socket.close()
        print("[Hardware Mock] Streaming finished. Server closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Hardware Streaming Mock")
    parser.add_argument("runs", metavar="N", type=int, nargs="+", help="Run numbers (e.g., 4 14)")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    
    start_server(args.host, args.port, args.subject, args.runs)