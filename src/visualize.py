import argparse
import matplotlib.pyplot as plt
from bci.eegbci import fetch_raw_eegbci, bandpass_filter_raw, L_FREQ, H_FREQ

def visualize_eeg(subject, runs):
    print(f"Fetching data for Subject {subject}, Runs {runs}...")
    try:
        raw = fetch_raw_eegbci(subject, runs)
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
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-109)")
    parser.add_argument("runs", metavar="N", type=int, nargs="+", help="Run numbers (e.g., 4 8 12)")
    args = parser.parse_args()
    
    visualize_eeg(args.subject, args.runs)