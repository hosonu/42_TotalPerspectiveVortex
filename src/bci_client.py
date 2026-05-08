import sys
import socket
import pickle
import struct
import joblib
import argparse
from pathlib import Path
from mybci import get_model_file

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def start_client(host, port, use_bonus, runs):
    model_file = get_model_file(use_bonus)
    model_path = Path(model_file)
    if not model_path.exists():
        print(
            f"Error: Model file '{model_file}' not found. Please run 'train' first.")
        sys.exit(1)

    
    print(f"[AI Client] Loading BCI Pipeline from {model_file}...")
    pipeline = joblib.load(model_file)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[AI Client] Connecting to EEG Hardware at {host}:{port}...")

    try:
        client_socket.connect((host, port))
    except ConnectionRefusedError:
        print("Error: Could not connect. Is the sensor_server running ?")
        return

    print("[AI Client] Connected! Waiting for brain waves...\n")
    print("-" * 50)

    i = 0
    correct_predictions = 0

    try:
        while True:
            raw_msglen = recvall(client_socket, 4)
            if not raw_msglen:
                break
            if len(raw_msglen) < 4:
                print("\n[AI Client] Error: Incomplete message length received.")
                break
        
            msglen = struct.unpack('>I', raw_msglen)[0]
            if msglen == 0:
                print("\n[AI Client] End of stream signal received.")
                break

            data_bytes = recvall(client_socket, msglen)
            try:
                X_trial, truth = pickle.loads(data_bytes)
            except (pickle.UnpicklingError, EOFError, TypeError) as e:
                print(f"\n[AI Client] Error decoding payload: {e}")
                break

            prediction = pipeline.predict(X_trial)[0]
            is_equal = (prediction == truth)
            if is_equal:
                correct_predictions += 1

            pred_out = prediction + 1
            truth_out = truth + 1
            print(f"epoch {i:02d}: [{pred_out}] [{truth_out}] {is_equal}")

            i += 1

    except KeyboardInterrupt:
        print("\n[AI Client] Stopped by user.")
    finally:
        client_socket.close()
        print("[AI Client] Connection closed.")

        if i > 0:  
            accuracy = correct_predictions / i
            print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI Real-time AI Client")
    parser.add_argument("runs", metavar="N", type=int, nargs="+", help="Run numbers (e.g., 4 14)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--bonus", action="store_true", help="Use the custom bonus classifier")
    args = parser.parse_args()
    
    start_client(args.host, args.port, args.bonus, args.runs)