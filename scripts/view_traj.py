import h5py
import sys
import numpy as np

def view_trajectory(filename, traj_idx):
    try:
        with h5py.File(filename, 'r') as f:
            traj_name = f"trajectory_{traj_idx}"
            if traj_name not in f:
                print(f"❌ Error: {traj_name} not found in {filename}")
                return

            grp = f[traj_name]
            print(f"=== Viewing {traj_name} in {filename} ===")
            
            # Determine length
            length = 0
            if "labels" in grp:
                length = len(grp["labels"])
            elif "camera_0" in grp:
                length = len(grp["camera_0"])
            
            # Columns to show
            data = {}
            
            # Labels
            if "labels" in grp:
                data["LABEL"] = grp["labels"][:]
            else:
                data["LABEL"] = ["-"] * length

            # Time/Timestamp
            if "timestamp" in grp:
                data["TIME"] = grp["timestamp"][:]
            elif "time" in grp:
                data["TIME"] = grp["time"][:]
            
            # ANSI Colors
            RED = '\033[91m'
            YELLOW = '\033[93m'
            GREEN = '\033[92m'
            RESET = '\033[0m'

            print(f"\nLegend: {GREEN}SAFE (0){RESET}, {RED}UNSAFE (1){RESET}, {YELLOW}WEAK UNSAFE (2){RESET}")
            print("-" * 60)
            print(f"{'RANGE':<20} | {'LABEL':<15} | {'DURATION':<10}")
            print("-" * 60)

            # Compressed Range View
            labels = data.get("LABEL", [0] * length)
            if len(labels) == 0:
                print("No frames found.")
                return

            start_idx = 0
            current_label = labels[0]
            
            for i in range(1, length):
                if labels[i] != current_label:
                    # End of current range
                    end_idx = i - 1
                    duration = end_idx - start_idx + 1
                    
                    # Format output
                    range_str = f"{start_idx} -> {end_idx}"
                    label_str = "SAFE"
                    color = GREEN
                    
                    if current_label == 1:
                        label_str = "UNSAFE"
                        color = RED
                    elif current_label == 2:
                        label_str = "WEAK UNSAFE"
                        color = YELLOW
                        
                    print(f"{color}{range_str:<20} | {label_str:<15} | {duration:<10}{RESET}")
                    
                    # Start new range
                    start_idx = i
                    current_label = labels[i]
            
            # Print last range
            end_idx = length - 1
            duration = end_idx - start_idx + 1
            range_str = f"{start_idx} -> {end_idx}"
            label_str = "SAFE"
            color = GREEN
            
            if current_label == 1:
                label_str = "UNSAFE"
                color = RED
            elif current_label == 2:
                label_str = "WEAK UNSAFE"
                color = YELLOW
                
            print(f"{color}{range_str:<20} | {label_str:<15} | {duration:<10}{RESET}")
            print("-" * 60)
            print(f"Total Frames: {length}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_traj.py <file.h5> [traj_index]")
    else:
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        view_trajectory(sys.argv[1], idx)