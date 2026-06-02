import socket
import pickle
import struct
import time
import argparse

from bci.epochs import build_epochs
from bci.data import epochs_to_Xy


def start_server(host, port, subject, runs, dataset_cfg=None):
    if dataset_cfg is None:
        dataset_cfg = {}
    print(f"Loading EEG data for Subject {subject} [{dataset_cfg.get('dataset', 'eegbci')}]...")

    epochs = build_epochs(subject, runs=runs, **dataset_cfg)
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
    parser.add_argument("runs", metavar="N", type=int, nargs="*",
                        help="Run numbers for eegbci (e.g., 4 14). Omit for bcic4_2a.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--dataset", default="eegbci", choices=["eegbci", "bcic4_2a"])
    parser.add_argument("--data-path", default=None, metavar="DIR",
                        help="Directory with .gdf files (required for bcic4_2a)")
    parser.add_argument("--classes", nargs=2, default=None, metavar="CLASS",
                        help="Two class names for bcic4_2a")
    args = parser.parse_args()

    dataset_cfg: dict = {"dataset": args.dataset}
    if args.data_path is not None:
        dataset_cfg["data_path"] = args.data_path
    if args.classes is not None:
        dataset_cfg["classes"] = args.classes

    start_server(args.host, args.port, args.subject, args.runs or None,
                 dataset_cfg=dataset_cfg)