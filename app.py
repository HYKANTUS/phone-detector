import time
import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "yolo11s.pt" 
VIDEO_PATH = "sample_video.mp4" 
TARGET_CLASSES = {"cell phone", "remote"} 
CONF_THRESHOLD = 0.35 
COOLDOWN_SEC = 15 
LATE_DROP_SEC = 0.25
STREAK_REQUIRED = 15 
MIN_BOX_AREA_RATIO = 0.01 
# ---------------

def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080  

SCREEN_W, SCREEN_H = get_screen_size()

model = YOLO(MODEL_PATH)
id2name = model.names
wanted_ids = {i for i, n in id2name.items() if n in TARGET_CLASSES}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
win_name = "YOLOv11 Realtime Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(win_name, 640, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

last_trigger = 0
streak_hits = 0 

def play_video_ffpy(path: str):
    ff_opts = {'out_format': 'rgb24', 'sync': 'audio', 'genpts': 1, 'framedrop': True, 'analyzeduration': 0, 'probesize': 32, 'fflags': 'nobuffer'}
    player = MediaPlayer(path, ff_opts=ff_opts)
    start_wall = None
    win = "Video"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    dummy = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    cv2.imshow(win, dummy)
    cv2.waitKey(1)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    video_width = SCREEN_W - 250 
    video_height = SCREEN_H + 100
    cv2.resizeWindow(win, video_width, video_height)
    cv2.moveWindow(win, (SCREEN_W - video_width) // 2, (SCREEN_H - video_height) // 2)

    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception: pass

    try:
        while True:
            frame, val = player.get_frame()
            if val == 'eof': break
            if frame is None:
                time.sleep(0.005)
                continue
            img, pts = frame
            if start_wall is None:
                start_wall = time.time() - (pts if pts is not None else 0.0)
            if pts is not None:
                target = start_wall + pts
                delay = target - time.time()
                if delay < -LATE_DROP_SEC: continue
                elif delay > 0: time.sleep(min(delay, 0.05))
            w, h = img.get_size()
            bytearray_img = img.to_bytearray()[0]
            frame_bgr = np.frombuffer(bytearray_img, dtype=np.uint8).reshape(h, w, 3)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            cv2.imshow(win, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        player.close_player()
        cv2.destroyWindow(win)

print("[INFO] Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret: break

    H, W = frame.shape[:2]
    frame_area = float(H * W)
    results = model(frame, verbose=False)[0]
    annotated = frame.copy()

    hit = False
    if results.boxes and len(results.boxes) > 0:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # ONLY process if it's in our TARGET_CLASSES and meets threshold
            # ONLY process if it's in our TARGET_CLASSES and meets threshold
            if cls_id in wanted_ids and conf >= CONF_THRESHOLD:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check size ratio
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / frame_area >= MIN_BOX_AREA_RATIO:
                    hit = True
                    
                    # --- MANUAL DRAWING (ONLY TARGETS) ---
                    # Draw Red Outline
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Create label with confidence percentage
                    # Example: "Phone 0.85" or "Phone 85%"
                    label = f"Phone {int(conf * 100)}%"
                    
                    # Draw the label background (makes text easier to read)
                    (l_w, l_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - 25), (x1 + l_w, y1), (0, 0, 255), -1)
                    
                    # Draw "Phone" Label + Confidence in White for contrast over the red box
                    cv2.putText(annotated, label, (x1, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- GRACE PERIOD STREAK LOGIC ---
    if hit:
        streak_hits += 1
    elif streak_hits > 0:
        streak_hits -= 1 # Decay instead of instant reset
    
    # Cap at STREAK_REQUIRED + 2 for the "safety buffer"
    streak_hits = min(streak_hits, STREAK_REQUIRED + 2)

    # --- STREAK BAR (Visualized) ---
    # We clip the visual bar at 100% (STREAK_REQUIRED)
    visual_streak = min(streak_hits, STREAK_REQUIRED)
    bar_width = int((visual_streak / STREAK_REQUIRED) * 200)
    cv2.rectangle(annotated, (50, 50), (250, 80), (50, 50, 50), -1)
    cv2.rectangle(annotated, (50, 50), (50 + bar_width, 80), (0, 0, 255), -1)

    # --- FLASHING TEXT LOGIC ---
    # Fast flash (12 toggles per second)
    is_flash_on = int(time.time() * 15) % 2 == 0
    if hit and is_flash_on:
        text = "!!!"
        font = cv2.FONT_HERSHEY_TRIPLEX 
        
        # 1. Force the text to fill 95% of the current width (W)
        target_width = int(W * 0.95)
        
        # Get base size at scale 1.0 to calculate the multiplier
        (base_w, base_h), _ = cv2.getTextSize(text, font, 1.0, 1)
        
        # Calculate the exact scale needed to hit target_width
        dynamic_scale = target_width / base_w
        
        # Calculate thickness relative to the scale so it stays bold
        dynamic_thickness = max(2, int(dynamic_scale * 2))
        
        # 2. Re-calculate final dimensions for centering
        (final_w, final_h), baseline = cv2.getTextSize(text, font, dynamic_scale, dynamic_thickness)
        
        text_x = int((W - final_w) / 2)
        text_y = int((H + final_h) / 2) # Centers vertically

        # 3. Draw with dynamic thickness
        # Shadow
        cv2.putText(annotated, text, (text_x + 3, text_y + 3), font, 
                    dynamic_scale, (0, 0, 0), dynamic_thickness + 4) 
        # Main Red Text
        cv2.putText(annotated, text, (text_x, text_y), font, 
                    dynamic_scale, (0, 0, 255), dynamic_thickness)

    # --- TRIGGER ---
    now = time.time()
    if streak_hits >= STREAK_REQUIRED and (now - last_trigger) > COOLDOWN_SEC:
        last_trigger = now
        streak_hits = 0  
        print("[EVENT] Phone detected → playing video.")
        play_video_ffpy(VIDEO_PATH)

    # Resize window to keep ratio
    rect_w = cv2.getWindowImageRect(win_name)[2]
    if rect_w > 0:
        cv2.resizeWindow(win_name, rect_w, int(rect_w * (H/W)))
    
    cv2.imshow(win_name, annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
