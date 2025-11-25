```markdown
# Automated Vehicle Trajectory Extraction from Drone Video  
*Intelligent Transportation Systems – Course Project (CP_B)*  

Welcome to the repository for **automated trajectory data extraction and analysis from drone videos using YOLOv8 and a custom multi-object tracker**.[attached_file:1][attached_file:2]

This README is structured as an **interactive guide**:  
- Start with the **Quick Start** if you just want to run the pipeline.  
- Use the **Guided Walkthrough** sections when you want to understand or customize each stage.  

---

## 1. Project Overview  

This project takes a **drone video of traffic**, automatically detects and tracks vehicles using **YOLOv8**, estimates a **real-world scale (pixels-per-metre)** using lane geometry or car size, and outputs per-vehicle **trajectories, speed, and acceleration**, along with an optional **visual overlay video** of tracks.[attached_file:1][attached_file:2]

**Core components:**[attached_file:2]  
- `vehicletrajectoriesauto.py` – end‑to‑end pipeline: detection, tracking, automatic scale estimation, and CSV export.  
- Post‑processing scripts – compute speed/acceleration, summarize trajectories, and visualize results.  
- Overlay renderer – write an output video with colored bounding boxes and trajectory trails for each tracked vehicle.  

---

## 2. Environment Setup (Step‑by‑Step)  

> Run these in a fresh Python 3 environment (e.g., Conda, venv, or Google Colab).[attached_file:2]

### 2.1. Install Core Dependencies  

```
pip install ultralytics opencv-python-headless pandas numpy tqdm
```

If you plan to run the **smoothness / evaluation** notebook cells:  

```
pip install trackeval matplotlib
```

> Note: `trackeval` may try to pin specific versions of NumPy / OpenCV; if you see version conflicts, prefer creating a dedicated environment or running evaluation in a separate environment / Colab runtime.[attached_file:2]

### 2.2. Recommended Python Versions  

- Python ≥ 3.9  
- GPU‑enabled PyTorch is recommended but **not mandatory**; YOLOv8 automatically uses CPU if GPU is not available.[attached_file:2]

---

## 3. Quick Start: One‑Command Trajectory Extraction  

This section assumes you have:  
- A drone video file (e.g., `content/drive/MyDrive/Omassery1.mp4`).  
- A YOLOv8 model file (e.g., `yolov8n.pt`, downloaded automatically by `ultralytics`).[attached_file:2]

### 3.1. Configure User Parameters  

In `vehicletrajectoriesauto.py`, set:[attached_file:1][attached_file:2]

```
# User PARAMETERS
VIDEOPATH = "content/drive/MyDrive/Omassery1.mp4"  # input drone video
OUTPUTCSV = "trajectories.csv"                     # aggregated output
OUTPUTDIR = "trajectories_per_vehicle"             # per-vehicle CSV folder
YOLOMODEL = "yolov8n.pt"                           # object detection model
CONFIDENCETHRESHOLD = 0.35                         # detection confidence
FRAMESKIP = 1                                      # process every k-th frame

ASSUMELANEWIDTHM = 3.5  # lane-based scale heuristic (metres)
ASSUMECARLENGTHM = 4.5  # car-size scale heuristic (metres)
```

### 3.2. Run the Main Script  

```
python vehicletrajectoriesauto.py
```

What this does, in sequence:[attached_file:1][attached_file:2]  

1. Loads YOLOv8 (`YOLOMODEL`).  
2. **Automatically estimates pixels-per-metre** using:
   - `ffprobe` metadata (if available), else  
   - **lane-based scale** from Hough line detection, else  
   - **car-size heuristic** based on typical car length.  
3. Runs detection + tracking for each frame (possibly skipping via `FRAMESKIP`).  
4. Writes:
   - `trajectories.csv` – one row per (vehicle, frame) with time, \((x,y)\), speed, acceleration, bounding box, and a flag `usedmetres` indicating whether positions are in metres or pixels.  
   - One CSV per vehicle in `OUTPUTDIR`.  

---

## 4. Interactive Guide: Core Modules  

### 4.1. Detection and Simple Tracking  

The pipeline uses **Ultralytics YOLOv8** for detection and a custom **SimpleTracker** with IoU‑based association and centroid buffering.[attached_file:2]

Key ideas:  
- Only **vehicle classes** (COCO: car=2, motorcycle=3, bus=5, truck=7) are considered for tracking and scale estimation.  
- Each tracked object gets a persistent `VehicleID`, a deque of centroid positions, and a `lost` counter to handle short occlusions.  

Important tracker parameters you can tune:[attached_file:2]  

```
class SimpleTracker:
    def __init__(self, maxlost=30, iouthreshold=0.3):
        ...
```

- `maxlost`: number of frames an object can be missing before being deregistered.  
- `iouthreshold`: minimum IoU for assignment; higher = stricter matching.  

---

### 4.2. Scale Estimation (Pixels-per-Metre)  

The script tries **three strategies** in order:[attached_file:1][attached_file:2]

1. **Metadata via `ffprobe`** (`try_ffprobe_for_scale`)  
   - Attempts to read altitude or similar tags to derive scale, if camera parameters are known externally.  
   - Returns metadata to the user but does **not** compute scale directly.  

2. **Lane-based scale** (`estimate_scale_by_lane`)  
   - Detects roughly parallel road edges using Canny + probabilistic HoughLinesP.  
   - Assumes a typical lane width (`ASSUMELANEWIDTHM`) and converts median line spacing to pixels-per-metre.  

3. **Car-size scale** (`estimate_scale_by_carsize`)  
   - Computes median bounding box length for common vehicle classes.  
   - Assumes typical car length (`ASSUMECARLENGTHM`) to derive pixels-per-metre.  

If all heuristics fail, the code falls back to **pixel units** (speeds in px/s, accelerations in px/s²) and sets `usedmetres=False`.[attached_file:1][attached_file:2]

---

### 4.3. Main Processing Loop  

The `process_video` function encapsulates detection, tracking, and trajectory logging.[attached_file:1][attached_file:2]

For every processed frame:  

- Run YOLO detection with confidence ≥ `CONFIDENCETHRESHOLD`.  
- Filter detections to vehicle classes.  
- Update the `SimpleTracker` with bounding boxes.  
- For each active object:
  - Compute centroid \((x,y)\) in pixels.  
  - Convert to metres if `pixelspermetre` is available.  
  - Estimate **instantaneous speed** (based on previous centroid) and **acceleration** (based on two past speeds).  
  - Append a row to `rows` with:
    - `VehicleID`, `FrameNo`, `Times`  
    - `Xm`/`Ym` (or `Xpx`/`Ypx`)  
    - `Speedmps`/`Speedpxps`, `Accmps2`/`Accpxps2`  
    - `usedmetres`, raw bounding box coordinates.  

At the end, a `pandas.DataFrame` is created, sorted by `VehicleID` and `FrameNo`, and saved to `OUTPUTCSV`, along with **per‑vehicle CSVs** in `OUTPUTDIR`.[attached_file:1][attached_file:2]

---

## 5. Post‑Processing: Speed & Acceleration (Interactive)  

If you want to **recompute or adjust speed/acceleration** based on a custom scale factor or corrected FPS:[attached_file:1][attached_file:2]

1. Load the raw `trajectories.csv`:  

   ```
   import pandas as pd
   import numpy as np

   df = pd.read_csv("trajectories.csv")

   # Ensure sorted by vehicle and time
   df = df.sort_values(by=["VehicleID", "FrameNo"])
   ```

2. Optionally override **scale factor** and **FPS**:  

   ```
   scalefactor = 0.05  # metres per pixel (example; replace with your calibration)
   fps = 30            # actual FPS of your video
   ```

3. Recompute `Times`, `Speedmps`, `Accmps2` for each `VehicleID` as in the notebook:[attached_file:1][attached_file:2]  

   - `Times` from `FrameNo / fps` if missing.  
   - Distance using \(\sqrt{\Delta x^2 + \Delta y^2}\).  
   - Speed = distance / \(\Delta t\).  
   - Acceleration = speed difference / \(\Delta t\).  

4. Save the updated CSV:  

   ```
   df.to_csv("trajectory_with_speed_accel.csv", index=False)
   ```

---

## 6. Visualization: Trajectories, Speed Plots & Overlay Video  

### 6.1. Trajectory and Speed Plots  

Example outline (as in the notebook):[attached_file:1][attached_file:2]  

```
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("trajectory_with_speed_accel.csv")

# XY trajectories
for vid in df["VehicleID"].unique():
    temp = df[df["VehicleID"] == vid]
    plt.plot(temp["Xm"], temp["Ym"], label=f"Vehicle {vid}")

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Vehicle Trajectories")
plt.legend()
plt.show()

# Speed vs Time
for vid in df["VehicleID"].unique():
    temp = df[df["VehicleID"] == vid]
    plt.plot(temp["Times"], temp["Speedmps"], label=f"Vehicle {vid}")

plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.title("Speed vs Time")
plt.legend()
plt.show()
```

---

### 6.2. Rendering an Overlay Video  

The overlay script replays the original video and draws:[attached_file:1][attached_file:2]  
- Colored bounding boxes per vehicle.  
- Vehicle ID labels.  
- Short “trajectory tails” (past `traillength` positions).  

Key user settings:[attached_file:1][attached_file:2]  

```
inputvideo = "content/drive/MyDrive/Omassery1.mp4"     # original video
trajectorycsv = "trajectory_with_speed_accel.csv"      # processed CSV
outputvideo = "overlay_trajectory.mp4"                 # output file
traillength = 15                                       # past points to show
frameskip = 1                                          # 1 = every frame
```

Each frame:  
- Filter `df` to current `FrameNo`.  
- Draw bounding box, ID, and trajectory trail for each vehicle.  
- Write to the output `VideoWriter`.  

---

## 7. Customization Cheatsheet  

- **Tuning detection speed/accuracy**  
  - Change `YOLOMODEL` (e.g., `yolov8n.pt` → `yolov8s.pt` for more accuracy).  
  - Increase `FRAMESKIP` to reduce processing load (e.g., `2` → every second frame).  

- **Switching to pixel units only**  
  - Set `ASSUMELANEWIDTHM` and `ASSUMECARLENGTHM` to `None` (or break the heuristic) and ignore `ppm` estimation, then all kinematics remain in pixel units with `usedmetres=False`.  

- **Filtering by vehicle type**  
  - Edit the class filter: `if cls in [2, 3, 5, 7]:` in the detection loop to include/exclude certain COCO categories.  

---

## 8. Files & Notebooks  

- **`ITS_CP_B.ipynb`**  
  - Structured notebook with code cells for:
    - Dependency installation (Ultralytics, TrackEval, etc.).  
    - The main `vehicletrajectoriesauto.py` logic.  
    - Post‑processing of trajectories, speed, and acceleration.  
    - Evaluation / smoothness plots and visualizations.[attached_file:2]  

- **`ITS_CP_B-Python-Codes.pdf`**  
  - Consolidated Python code listing for the full pipeline, including tracker, scale estimation, processing script, evaluation, and overlay rendering.[attached_file:1]  

---

## 9. Suggested Workflow (Interactive)  

1. **Prototype in Notebook (`ITS_CP_B.ipynb`)**  
   - Mount or upload input video.  
   - Run installation/code cells step‑wise and inspect intermediate outputs (`trajectories.csv`, plots).  

2. **Package as Script (`vehicletrajectoriesauto.py`)**  
   - Fix paths and parameters for batch processing.  
   - Run on multiple videos, storing results in structured folders.  

3. **Analyze & Visualize**  
   - Use `trajectory_with_speed_accel.csv` for ITS analyses (speed profiles, headways, lane‑level KPIs).  
   - Use overlay video for qualitative validation of tracking quality.  

4. **Refine**  
   - Adjust `ASSUMELANEWIDTHM`, `ASSUMECARLENGTHM`, `CONFIDENCETHRESHOLD`, `maxlost`, `iouthreshold` based on specific site geometry and video quality.  

---

## 10. Acknowledgements  

This implementation uses:  
- **Ultralytics YOLOv8** for object detection.[attached_file:2]  
- **OpenCV** for video IO, image processing, and drawing.[attached_file:1]  
- **Pandas / NumPy** for data handling and kinematic calculations.[attached_file:1]  
