import argparse
import matplotlib.pyplot as plt
from bci.eegbci import fetch_raw_eegbci, bandpass_filter_raw, L_FREQ, H_FREQ


def _load_raw(subject, runs, dataset, data_path):
    if dataset == "bcic4_2a":
        if data_path is None:
            raise ValueError("--data-path is required for dataset 'bcic4_2a'")
        from bci.datasets.bcic4_2a import fetch_raw
        return fetch_raw(subject, data_path=data_path)
    return fetch_raw_eegbci(subject, runs)


def visualize_eeg(subject, runs, dataset="eegbci", data_path=None):
    print(f"Fetching data for Subject {subject} [{dataset}]...")
    try:
        raw = _load_raw(subject, runs, dataset, data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        print("Plotting Raw Data...")
        fig_raw_psd = raw.compute_psd(fmax=50).plot(show=False)
        fig_raw_psd.suptitle(f"Raw PSD (Subject {subject}, Runs {runs})")

        fig_raw_time = raw.plot(duration=5, n_channels=20, show=False, title="Raw EEG Data")

        print(f"Applying Bandpass Filter ({L_FREQ}-{H_FREQ} Hz)...")
        raw_filtered = raw.copy()
        bandpass_filter_raw(raw_filtered)

        print("Plotting Filtered Data...")
        fig_filt_psd = raw_filtered.compute_psd(fmax=50).plot(show=False)
        fig_filt_psd.suptitle(f"Filtered PSD ({L_FREQ}-{H_FREQ} Hz)")

        fig_filt_time = raw_filtered.plot(duration=5, n_channels=20, show=False, title="Filtered EEG Data")

        plt.show()

    except KeyboardInterrupt:
        print("\n[Visualize] Stopped by user.")
    except Exception as e:
        print(f"\n[Visualize] Error during plotting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize raw and filtered EEG data")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID")
    parser.add_argument("runs", metavar="N", type=int, nargs="*",
                        help="Run numbers for eegbci (e.g., 4 8 12). Omit for bcic4_2a.")
    parser.add_argument("--dataset", default="eegbci", choices=["eegbci", "bcic4_2a"])
    parser.add_argument("--data-path", default=None, metavar="DIR",
                        help="Directory with .gdf files (required for bcic4_2a)")
    args = parser.parse_args()

    visualize_eeg(args.subject, args.runs or None,
                  dataset=args.dataset, data_path=args.data_path)