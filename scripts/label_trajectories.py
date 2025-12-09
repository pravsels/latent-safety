#!/usr/bin/env python3
"""
HDF5 Trajectory Safety Labeling Tool
------------------------------------
A simple tool to label trajectories as safe/unsafe in HDF5 format.

Controls:
  SPACE      : Play/Pause
  LEFT/RIGHT : Previous/Next frame
  UP/DOWN    : Previous/Next trajectory
  U          : Toggle Unsafe Region (Start... End)
  W          : Toggle Weak Unsafe Region (Start... End)
  Z          : Undo last action
  C          : Clear all labels
  Q          : Save & Quit

Usage:
  python scripts/label_trajectories.py --hdf5 data.h5 --session session_name.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np

# --- Constants ---
WINDOW_NAME = "Safety Labeling Tool"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_UNSAFE = (0, 0, 255)    # Red
COLOR_WEAK_UNSAFE = (0, 255, 255)    # Yellow
COLOR_TEXT = (255, 255, 255)  # White

class LabelingSession:
    def __init__(self, hdf5_path: str, session_path: str):
        self.hdf5_path = Path(hdf5_path)
        self.session_path = Path(session_path)
        self.data = self._load_session()
        
    def _load_session(self) -> Dict:
        if self.session_path.exists():
            with open(self.session_path, 'r') as f:
                print(f"📂 Loaded session: {self.session_path}")
                return json.load(f)
        return {
            "hdf5_file": str(self.hdf5_path),
            "current_trajectory": 0
        }
    
    def save(self):
        with open(self.session_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"💾 Session saved to {self.session_path}")

    @property
    def current_traj_idx(self) -> int:
        return self.data["current_trajectory"]
    
    @current_traj_idx.setter
    def current_traj_idx(self, value: int):
        self.data["current_trajectory"] = value

class TrajectoryLabeler:
    def __init__(self, hdf5_path: str, session: LabelingSession):
        self.hdf5_path = hdf5_path
        self.session = session
        self.hf = h5py.File(hdf5_path, 'r+')
        
        # Get sorted list of trajectory keys
        def safe_int_key(x):
            try:
                return int(x.split('_')[1])
            except (ValueError, IndexError):
                return float('inf')  # Put invalid keys at the end
        
        self.traj_keys = sorted(
            [k for k in self.hf.keys() if k.startswith("trajectory_")],
            key=safe_int_key
        )
        self.num_trajectories = len(self.traj_keys)
        
        # State
        self.current_frame = 0
        self.playing = False
        self.mark_start_frame: Optional[int] = None
        self.mark_label_type: int = 1  # 1 for unsafe, 2 for weak unsafe
        self.labels: np.ndarray = None
        self.history: List[np.ndarray] = []  # For undo
        
        # Cache current trajectory data
        self.current_images_0 = None
        self.current_images_1 = None
        self.total_frames = 0
        
        self.load_trajectory(self.session.current_traj_idx)

    def load_trajectory(self, idx: int):
        if idx < 0 or idx >= self.num_trajectories:
            return
            
        self.session.current_traj_idx = idx
        traj_name = self.traj_keys[idx]
        grp = self.hf[traj_name]
        
        print(f"Loading {traj_name}...")
        
        # Load images (assuming they fit in memory for smooth playback)
        if "camera_0" not in grp or "camera_1" not in grp:
            print(f"Error: Missing camera_0 or camera_1 in {traj_name}")
            return
            
        self.current_images_0 = grp["camera_0"][:]
        self.current_images_1 = grp["camera_1"][:]
        self.total_frames = len(self.current_images_0)
        
        if self.total_frames == 0:
            print(f"Warning: {traj_name} has no frames")
            return
        
        # Load or init labels
        if "labels" in grp:
            self.labels = grp["labels"][:]
            if len(self.labels) != self.total_frames:
                print(f"Warning: Label length mismatch, reinitializing labels")
                self.labels = np.zeros(self.total_frames, dtype=np.int8)
        else:
            self.labels = np.zeros(self.total_frames, dtype=np.int8)
            
        self.current_frame = 0
        self.mark_start_frame = None
        self.mark_label_type = 1
        self.history = []
        self.playing = False
        
        # Navigation acceleration state
        self.last_nav_time = 0.0
        self.nav_speed = 1.0
        self.last_nav_key = None

    def save_labels(self):
        traj_name = self.traj_keys[self.session.current_traj_idx]
        grp = self.hf[traj_name]
        
        if "labels" in grp:
            del grp["labels"]
        grp.create_dataset("labels", data=self.labels)
        grp.attrs["labeled"] = True
        grp.attrs["labeled_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.hf.flush()
        self.session.save()

    def push_history(self):
        self.history.append(self.labels.copy())
        if len(self.history) > 10:
            self.history.pop(0)

    def undo(self):
        if self.history:
            self.labels = self.history.pop()
            print("Undo!")
        else:
            print("Nothing to undo")

    def toggle_unsafe_region(self):
        if self.mark_start_frame is None:
            # Start marking unsafe
            self.mark_start_frame = self.current_frame
            self.mark_label_type = 1
        else:
            # If already marking, cancel previous mark and start new one
            if self.mark_label_type != 1:
                self.mark_start_frame = self.current_frame
                self.mark_label_type = 1
            else:
                # End marking
                self.push_history()
                start = min(self.mark_start_frame, self.current_frame)
                end = max(self.mark_start_frame, self.current_frame)
                
                # Mark range as unsafe (1)
                self.labels[start:end+1] = 1
                self.mark_start_frame = None

    def toggle_weak_unsafe_region(self):
        if self.mark_start_frame is None:
            # Start marking weak unsafe
            self.mark_start_frame = self.current_frame
            self.mark_label_type = 2
        else:
            # If already marking, cancel previous mark and start new one
            if self.mark_label_type != 2:
                self.mark_start_frame = self.current_frame
                self.mark_label_type = 2
            else:
                # End marking
                self.push_history()
                start = min(self.mark_start_frame, self.current_frame)
                end = max(self.mark_start_frame, self.current_frame)
                
                # Mark range as weak unsafe (2)
                self.labels[start:end+1] = 2
                self.mark_start_frame = None

    def clear_labels(self):
        self.push_history()
        self.labels[:] = 0
        self.mark_start_frame = None
        self.mark_label_type = 1
        print("Cleared all labels")

    def render(self):
        # Get frames
        img0 = self.current_images_0[self.current_frame]
        img1 = self.current_images_1[self.current_frame]
        
        # Ensure BGR for OpenCV
        if img0.shape[-1] == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            
        # Concatenate side-by-side
        vis = np.hstack([img0, img1])
        
        # Overlay Info
        h, w = vis.shape[:2]
        
        # Status
        if self.total_frames == 0 or self.labels is None:
            return vis
            
        label_val = self.labels[self.current_frame] if self.current_frame < len(self.labels) else 0
        if label_val == 0:
            status_color = COLOR_TEXT  # Default white for safe
            status_text = "SAFE"
        elif label_val == 1:
            status_color = COLOR_UNSAFE
            status_text = "UNSAFE"
        else:  # label_val == 2
            status_color = COLOR_WEAK_UNSAFE
            status_text = "WEAK UNSAFE"
        
        if self.mark_start_frame is not None:
            mark_type = "UNSAFE" if self.mark_label_type == 1 else "WEAK UNSAFE"
            status_text = f"MARKING {mark_type} {self.mark_start_frame}→{self.current_frame}"
            status_color = COLOR_WEAK_UNSAFE if self.mark_label_type == 2 else COLOR_UNSAFE
            
        # Minimal top bar
        cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
        
        # Left: Trajectory
        traj_info = f"Traj: {self.session.current_traj_idx + 1}/{self.num_trajectories}"
        cv2.putText(vis, traj_info, (10, 20), FONT, 0.5, COLOR_TEXT, 1)
        
        # Center: Frame
        frame_info = f"Frame: {self.current_frame + 1}/{self.total_frames}"
        cv2.putText(vis, frame_info, (w//2 - 60, 20), FONT, 0.5, status_color, 1)
        
        # Right: Status
        cv2.putText(vis, status_text, (w - 120, 20), FONT, 0.5, status_color, 1)
        
        # Timeline at bottom
        bar_h = 15
        cv2.rectangle(vis, (0, h-bar_h), (w, h), (30, 30, 30), -1)
        
        # Draw unsafe and weak unsafe regions on timeline
        if self.total_frames == 0:
            return vis
        scale = w / self.total_frames
        unsafe_indices = np.where(self.labels == 1)[0]
        weak_unsafe_indices = np.where(self.labels == 2)[0]
        
        for idx in unsafe_indices:
            x = int(idx * scale)
            cv2.line(vis, (x, h-bar_h), (x, h), COLOR_UNSAFE, 1)
        
        for idx in weak_unsafe_indices:
            x = int(idx * scale)
            cv2.line(vis, (x, h-bar_h), (x, h), COLOR_WEAK_UNSAFE, 1)
            
        # Draw current position cursor
        cursor_x = int(self.current_frame * scale)
        cv2.line(vis, (cursor_x, h-bar_h), (cursor_x, h), (255, 255, 255), 2)
        
        # Draw mark start if active
        if self.mark_start_frame is not None:
            mark_x = int(self.mark_start_frame * scale)
            mark_color = COLOR_WEAK_UNSAFE if self.mark_label_type == 2 else COLOR_UNSAFE
            cv2.line(vis, (mark_x, h-bar_h), (mark_x, h), mark_color, 2)
            # Highlight region between mark and cursor
            x1 = min(mark_x, cursor_x)
            x2 = max(mark_x, cursor_x)
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, h-bar_h), (x2, h), mark_color, -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        return vis

    def show_loading_message(self, idx: int):
        vis = self.render()
        h, w = vis.shape[:2]
        
        # Darken background
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        # Text
        text = f"Loading Trajectory {idx + 1}/{self.num_trajectories}..."
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
        
        x = (w - text_w) // 2
        y = (h + text_h) // 2
        
        cv2.putText(vis, text, (x, y), FONT, font_scale, (255, 255, 255), thickness)
        cv2.imshow(WINDOW_NAME, vis)
        cv2.waitKey(1)

    def run(self):
        print("\n=== Controls ===")
        print("SPACE      : Play/Pause")
        print("LEFT/RIGHT : Prev/Next frame")
        print("UP/DOWN    : Next/Prev trajectory")
        print("U          : Toggle Unsafe Region (Start... End)")
        print("W          : Toggle Weak Unsafe Region (Start... End)")
        print("Z          : Undo")
        print("C          : Clear all")
        print("Q          : Save & Quit")
        print("================")
        
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Set window to fullscreen by default
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while True:
            vis = self.render()
            cv2.imshow(WINDOW_NAME, vis)
            
            key = cv2.waitKey(30 if self.playing else 0) & 0xFF
            
            # Controls
            if key == ord('q'):
                self.save_labels()
                break
            elif key == ord(' '):
                self.playing = not self.playing
            
            # Navigation with Acceleration
            elif key in [81, ord('a'), 83, ord('d')]: # Left/Right
                now = time.time()
                is_left = (key == 81 or key == ord('a'))
                
                # Check for consecutive press (hold)
                # Group keys: Left/A are same "direction", Right/D are same "direction"
                # But for simplicity, just check if key matches last key
                if key == self.last_nav_key and (now - self.last_nav_time < 0.2):
                    # Accelerate
                    self.nav_speed = min(self.nav_speed + 1, 10)
                else:
                    # Reset
                    self.nav_speed = 1
                
                self.last_nav_key = key
                self.last_nav_time = now
                
                step = int(self.nav_speed)
                if is_left:
                    self.current_frame = max(0, self.current_frame - step)
                else:
                    self.current_frame = min(self.total_frames - 1, self.current_frame + step)
                    
            elif key == 82: # Up arrow (Next Traj)
                next_idx = self.session.current_traj_idx + 1
                if next_idx < self.num_trajectories:
                    self.show_loading_message(next_idx)
                    self.save_labels()
                    self.load_trajectory(next_idx)
            elif key == 84 or key == ord('s'): # Down arrow or 's' (Prev Traj)
                next_idx = self.session.current_traj_idx - 1
                if next_idx >= 0:
                    self.show_loading_message(next_idx)
                    self.save_labels()
                    self.load_trajectory(next_idx)
            elif key == ord('u'):
                self.toggle_unsafe_region()
            elif key == ord('w'):
                self.toggle_weak_unsafe_region()
            elif key == ord('z'):
                self.undo()
            elif key == ord('c'):
                self.clear_labels()
            
            # Auto-advance if playing
            if self.playing:
                if self.current_frame < self.total_frames - 1:
                    self.current_frame += 1
                else:
                    self.playing = False
                    
        self.hf.close()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--session", required=False, help="Path to session JSON file (auto-generated if not provided)")
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5):
        print(f"❌ Error: File not found: {args.hdf5}")
        sys.exit(1)
    
    # Auto-generate session filename if not provided
    if args.session is None:
        hdf5_path = Path(args.hdf5)
        session_path = hdf5_path.with_suffix('.session.json')
        print(f"📝 No session file provided, using: {session_path}")
    else:
        session_path = args.session
        
    session = LabelingSession(args.hdf5, session_path)
    labeler = TrajectoryLabeler(args.hdf5, session)
    labeler.run()

if __name__ == "__main__":
    main()
