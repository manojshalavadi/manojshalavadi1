import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from functools import lru_cache

# -------------------------------
# SETTINGS
# -------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

LINE_Y = 180

CONFIDENCE_THRESHOLD = 0.55

# -------------------------------
# LOAD MODEL ONLY ONCE
# -------------------------------
@lru_cache(maxsize=1)
def load_model():
    return YOLO("yolov8n.pt")

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def process_video(video_path):

    # -------------------------------
    # VIDEO SOURCE
    # -------------------------------
    if video_path == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    # -------------------------------
    # CHECK VIDEO
    # -------------------------------
    if not cap.isOpened():
        raise Exception("Unable to open video source")

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = load_model()

    # -------------------------------
    # TRACKER
    # -------------------------------
    tracker = DeepSort(
        max_age=40,
        n_init=3,
        max_cosine_distance=0.2,
        nn_budget=100
    )

    # -------------------------------
    # VARIABLES
    # -------------------------------
    previous_positions = {}

    counted_in = set()
    counted_out = set()

    total_detected_ids = set()

    in_count = 0
    out_count = 0

    frame_count = 0
    prev_time = time.time()

    # -------------------------------
    # HEATMAP
    # -------------------------------
    heatmap = np.zeros(
        (FRAME_HEIGHT, FRAME_WIDTH),
        dtype=np.float32
    )

    # -------------------------------
    # LOOP
    # -------------------------------
    while True:

        ret, frame = cap.read()

        # VIDEO END
        if not ret:
            break

        # -------------------------------
        # RESIZE
        # -------------------------------
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # -------------------------------
        # FRAME SKIP
        # -------------------------------
        frame_count += 1

        if frame_count % 2 != 0:
            continue

        # -------------------------------
        # YOLO DETECTION
        # -------------------------------
        results = model(
            frame,
            imgsz=416,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],   # PERSON ONLY
            verbose=False
        )

        detections = []

        # -------------------------------
        # FILTER DETECTIONS
        # -------------------------------
        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # PERSON ONLY
                if cls != 0:
                    continue

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                w = x2 - x1
                h = y2 - y1

                # REMOVE SMALL OBJECTS
                area = w * h

                if area < 2500:
                    continue

                # REMOVE INVALID SHAPES
                ratio = w / h

                if ratio > 0.9 or ratio < 0.2:
                    continue

                detections.append(
                    ([x1, y1, w, h], conf, "person")
                )

        # -------------------------------
        # TRACKING
        # -------------------------------
        tracks = tracker.update_tracks(
            detections,
            frame=frame
        )

        current_ids = set()

        # -------------------------------
        # COUNTING LINE
        # -------------------------------
        cv2.line(
            frame,
            (0, LINE_Y),
            (FRAME_WIDTH, LINE_Y),
            (0, 255, 255),
            3
        )

        # -------------------------------
        # TRACK LOOP
        # -------------------------------
        for track in tracks:

            if not track.is_confirmed():
                continue

            # SAFE TRACK BOX
            try:
                l, t, r, b = map(int, track.to_ltrb())
            except:
                continue

            track_id = track.track_id

            w = r - l
            h = b - t

            # REMOVE BAD BOXES
            if h < 70:
                continue

            # CENTER
            cX = int((l + r) / 2)
            cY = int((t + b) / 2)

            current_ids.add(track_id)

            # STORE TOTAL UNIQUE PERSONS
            total_detected_ids.add(track_id)

            # -------------------------------
            # HEATMAP
            # -------------------------------
            if 0 <= cX < FRAME_WIDTH and 0 <= cY < FRAME_HEIGHT:
                heatmap[cY, cX] += 1

            # -------------------------------
            # COUNTING LOGIC
            # -------------------------------
            prevY = previous_positions.get(
                track_id,
                None
            )

            if prevY is not None:

                # TOP → BOTTOM = IN
                if prevY < LINE_Y and cY >= LINE_Y:

                    if track_id not in counted_in:

                        in_count += 1
                        counted_in.add(track_id)

                # BOTTOM → TOP = OUT
                elif prevY > LINE_Y and cY <= LINE_Y:

                    if track_id not in counted_out:

                        out_count += 1
                        counted_out.add(track_id)

            previous_positions[track_id] = cY

            # -------------------------------
            # DRAW BOX
            # -------------------------------
            cv2.rectangle(
                frame,
                (l, t),
                (r, b),
                (0, 255, 0),
                2
            )

            # LABEL
            cv2.putText(
                frame,
                f"ID {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # CENTER DOT
            cv2.circle(
                frame,
                (cX, cY),
                4,
                (0, 0, 255),
                -1
            )

        # -------------------------------
        # LIVE HUMAN COUNT
        # -------------------------------
        live = len(current_ids)

        # -------------------------------
        # CLEAN OLD TRACKS
        # -------------------------------
        inactive_ids = (
            set(previous_positions.keys())
            - current_ids
        )

        for old_id in inactive_ids:
            previous_positions.pop(old_id, None)

        # -------------------------------
        # RESET HEATMAP
        # -------------------------------
        if frame_count % 200 == 0:
            heatmap[:] = 0

        # -------------------------------
        # HEATMAP VISUAL
        # -------------------------------
        heatmap_blur = cv2.GaussianBlur(
            heatmap,
            (21, 21),
            0
        )

        heatmap_norm = cv2.normalize(
            heatmap_blur,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        heatmap_color = cv2.applyColorMap(
            heatmap_norm.astype(np.uint8),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            frame,
            0.75,
            heatmap_color,
            0.25,
            0
        )

        # -------------------------------
        # FPS
        # -------------------------------
        current_time = time.time()

        fps = int(
            1 / (current_time - prev_time)
        ) if (current_time - prev_time) > 0 else 0

        prev_time = current_time

        # -------------------------------
        # DISPLAY TEXT
        # -------------------------------
        cv2.putText(
            overlay,
            f"IN: {in_count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            f"OUT: {out_count}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            overlay,
            f"LIVE: {live}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            f"TOTAL: {len(total_detected_ids)}",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 165, 0),
            2
        )

        cv2.putText(
            overlay,
            f"FPS: {fps}",
            (15, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # -------------------------------
        # RETURN
        # -------------------------------
        yield (
            overlay,
            in_count,
            out_count,
            live,
            len(total_detected_ids)
        )

    # -------------------------------
    # RELEASE
    # -------------------------------
    cap.release()