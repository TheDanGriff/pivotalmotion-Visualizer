import os
import csv
import cv2
from ultralytics import YOLO

def main():
    # ---------------------------------------------------------------------
    # 1. Load YOLOv8 model
    # ---------------------------------------------------------------------
    model_path = r"C:\Users\dgriff06\Downloads\best.pt"
    model = YOLO(model_path)

    # ---------------------------------------------------------------------
    # 2. Configure source video and output folder
    # ---------------------------------------------------------------------
    video_path = r"C:\Users\dgriff06\Downloads\3D_Basketball\Pivotal\Project\Pivotal-Motion-Main-Processing\Users\dgriff06@syr.edu\3b64f258-460b-4b03-ad51-a21465ddaf7c\segments\segment_001.mp4"
    # Where tracked results (annotated video) will be saved
    # YOLOv8 automatically creates subfolders like runs/track/exp if not specified
    output_dir = "runs/track/exp"

    # ---------------------------------------------------------------------
    # 3. Run model tracking on the video
    #    show=True   -> displays window with detections
    #    save=True   -> saves annotated result video to output_dir
    #    tracker='bytetrack.yaml' -> use built-in ByteTrack tracker
    # ---------------------------------------------------------------------
    results = model.track(
        source=video_path,
        show=True,
        save=True,
        project=output_dir,
        name="tracked",
        tracker="bytetrack.yaml"  # You can also provide your own tracker config
    )

    # ---------------------------------------------------------------------
    # 4. Save the tracking data to a CSV
    #    YOLOv8’s model.track() returns a list of Results objects, each
    #    containing data (boxes, masks, etc.) for each frame
    # ---------------------------------------------------------------------
    csv_file_path = os.path.join(output_dir, "detections.csv")

    # Open a CSV file to write all detections/tracks
    with open(csv_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["frame", "track_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"])
        
        # Loop through each frame's results
        for frame_idx, frame_result in enumerate(results):
            # Each 'frame_result' is a ultralytics.yolo.engine.results.Results object
            boxes = frame_result.boxes  # Boxes object
            if boxes is None:
                continue

            # boxes.xyxy -> (N, 4) [xmin, ymin, xmax, ymax]
            # boxes.id   -> (N,) track IDs for each box (if tracking)
            # boxes.cls  -> (N,) class IDs
            # boxes.conf -> (N,) confidence
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()  # e.g. [xmin, ymin, xmax, ymax]
                track_id = boxes.id[i].item() if boxes.id is not None else -1
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                row = [
                    frame_idx,
                    track_id,
                    cls_id,
                    f"{conf:.4f}",
                    f"{xyxy[0]:.2f}",
                    f"{xyxy[1]:.2f}",
                    f"{xyxy[2]:.2f}",
                    f"{xyxy[3]:.2f}"
                ]
                writer.writerow(row)

    print(f"Tracking completed. Annotated video saved to: {os.path.abspath(output_dir)}")
    print(f"Detections/Tracking data saved to CSV: {os.path.abspath(csv_file_path)}")

    # ---------------------------------------------------------------------
    # 5. (Optional) Validate the CSV by reading it back and printing stats
    # ---------------------------------------------------------------------
    validate_csv(csv_file_path)

def validate_csv(csv_file_path):
    """Simple function to read back the CSV file and validate data."""
    import pandas as pd
    
    if not os.path.exists(csv_file_path):
        print("CSV file not found for validation.")
        return

    df = pd.read_csv(csv_file_path)
    print("\n--- CSV Validation ---")
    print(df.head())
    print(f"\nTotal detections in CSV: {len(df)}")
    print(f"Unique frames: {df['frame'].nunique()}")
    if 'track_id' in df.columns:
        print(f"Unique track IDs: {df['track_id'].nunique()}")

if __name__ == "__main__":
    main()
