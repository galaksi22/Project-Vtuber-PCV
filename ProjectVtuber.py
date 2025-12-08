import cv2
import mediapipe as mp
import numpy as np
import os
import json

# ================= KONFIGURASI =================
SCALE = 0.35 
ASSET_DIR = "assets" 
CONFIG_FILE = "vtuber_config.json"
BG_FILENAME = "Background.png"

# FORMAT DATA: [X, Y, SCALE_X, SCALE_Y, ROTATION]
DEFAULT_OFFSETS = {
    "front": { "left_eye": [626, 782, 100, 100, 0], "right_eye": [1111, 782, 100, 100, 0], "mouth": [971, 1180, 100, 100, 0] },
    "left":  { "left_eye": [450, 780, 85, 100, 0],  "right_eye": [950, 780, 100, 100, 0],  "mouth": [800, 1180, 100, 100, 0] },
    "right": { "left_eye": [850, 780, 100, 100, 0], "right_eye": [1350, 780, 85, 100, 0], "mouth": [1100, 1180, 100, 100, 0] },
    "up":    { "left_eye": [680, 650, 100, 100, 0], "right_eye": [1080, 650, 100, 100, 0], "mouth": [970, 1050, 100, 100, 0] },
    "down":  { "left_eye": [680, 900, 100, 100, 0], "right_eye": [1080, 900, 100, 100, 0], "mouth": [970, 1300, 100, 100, 0] }
}

MOUTH_TWEAKS = {"idle": (0, 0), "talk": (0, 0), "open": (0, 0)}

# ================= UTILITIES =================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                if len(data["front"]["left_eye"]) == 5:
                    print(f"[INFO] Config loaded from {CONFIG_FILE}")
                    return data
        except: pass
    return DEFAULT_OFFSETS

def save_config(data):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Tersimpan ke {CONFIG_FILE}!")
    except Exception as e:
        print(f"[ERROR] Gagal save: {e}")

def nothing(x): pass

# --- FUNGSI RESIZE (FIXED CHANNEL ERROR) ---
def resize_keep_aspect_ratio(image, target_w, target_h):
    """
    Mengubah ukuran gambar agar pas di jendela TANPA merusak rasio (Letterbox).
    [FIX]: Otomatis menyesuaikan channel (BGR/BGRA) agar tidak crash.
    """
    h, w = image.shape[:2]
    if target_w <= 0 or target_h <= 0: return image
    
    # Deteksi jumlah channel (3 atau 4)
    channels = 3
    if len(image.shape) > 2:
        channels = image.shape[2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = max(1, new_w); new_h = max(1, new_h)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Buat kanvas dengan jumlah channel yang sama dengan input
    canvas = np.zeros((target_h, target_w, channels), dtype=np.uint8)
    
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    y_end = min(y_offset + new_h, target_h)
    x_end = min(x_offset + new_w, target_w)
    
    canvas[y_offset:y_end, x_offset:x_end] = resized[0:y_end-y_offset, 0:x_end-x_offset]
    
    return canvas

OFFSETS = load_config()

# ================= CLASSES =================
class VTuberRenderer:
    def __init__(self):
        self.assets = {}
        self.bg_color = (0, 255, 0) 
        self.bg_image = None        
        self.load_assets()

    def load_img(self, filename):
        path = os.path.join(ASSET_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return np.zeros((10, 10, 4), dtype=np.uint8)
        if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        new_w = int(img.shape[1] * SCALE)
        new_h = int(img.shape[0] * SCALE)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def load_assets(self):
        possible_bgs = ["Background.png", "Background.jpg", "Background.jpeg"]
        self.bg_image = None
        for bg_name in possible_bgs:
            bg_path = os.path.join(ASSET_DIR, bg_name)
            if os.path.exists(bg_path):
                self.bg_image = cv2.imread(bg_path); break
        
        self.assets = {
            "base": {
                "front": self.load_img("Muka_Depan.png"),
                "left":  self.load_img("Muka_Kiri.png"),
                "right": self.load_img("Muka_Kanan.png"),
                "up":    self.load_img("Muka_Atas.png"),
                "down":  self.load_img("muka_bawah.png"), 
            },
            "eyes_left": { 
                "normal": self.load_img("Mata_Kanan_3.png"),
                "squint": self.load_img("Mata_Kanan_2.png"),
                "blink":  self.load_img("Mata_Kanan_1.png"),
            },
            "eyes_right": {
                "normal": self.load_img("Mata_Kiri_3.png"),  
                "squint": self.load_img("Mata_Kiri_2.png"), 
                "blink":  self.load_img("Mata_Kiri_1.png"),  
            },
            "mouth": {
                "idle": self.load_img("Mulut_1.png"), 
                "talk": self.load_img("Mulut_2.png"), 
                "open": self.load_img("Mulut_3.png"), 
            }
        }

    def transform_and_overlay(self, bg, fg, params):
        x, y, scale_x_p, scale_y_p, rot_deg = params
        sx_factor = scale_x_p / 100.0; sy_factor = scale_y_p / 100.0
        if sx_factor <= 0.05: sx_factor = 0.05
        if sy_factor <= 0.05: sy_factor = 0.05
        
        h_orig, w_orig = fg.shape[:2]
        new_w = int(w_orig * sx_factor); new_h = int(h_orig * sy_factor)
        fg_resized = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if rot_deg != 0:
            h, w = fg_resized.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
            cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
            new_w_bound = int((h * sin) + (w * cos))
            new_h_bound = int((h * cos) + (w * sin))
            M[0, 2] += (new_w_bound / 2) - center[0]
            M[1, 2] += (new_h_bound / 2) - center[1]
            fg_final = cv2.warpAffine(fg_resized, M, (new_w_bound, new_h_bound), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        else:
            fg_final = fg_resized

        h_fg, w_fg = fg_final.shape[:2]
        x_pos = int(x * SCALE) - (w_fg // 2)
        y_pos = int(y * SCALE) - (h_fg // 2)
        return self.overlay_pixel(bg, fg_final, x_pos, y_pos)

    def overlay_pixel(self, bg, fg, x, y):
        h_fg, w_fg = fg.shape[:2]; h_bg, w_bg = bg.shape[:2]
        if x >= w_bg or y >= h_bg: return bg
        if x + w_fg < 0 or y + h_fg < 0: return bg
        x_start = max(0, x); y_start = max(0, y)
        x_end = min(w_bg, x + w_fg); y_end = min(h_bg, y + h_fg)
        fg_crop = fg[y_start-y:y_end-y, x_start-x:x_end-x]
        bg_crop = bg[y_start:y_end, x_start:x_end]
        alpha = fg_crop[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        composite = (fg_crop[:, :, :3] * alpha) + (bg_crop[:, :, :3] * (1.0 - alpha))
        bg[y_start:y_end, x_start:x_end, :3] = composite.astype(np.uint8)
        return bg

    def render(self, state):
        pose = state["pose"]
        if pose not in self.assets["base"]: pose = "front"
        
        # Render BG
        face_img = self.assets["base"][pose]
        base_h, base_w = face_img.shape[:2]
        if self.bg_image is not None:
            canvas = cv2.resize(self.bg_image, (base_w, base_h))
            if canvas.shape[2] == 3: canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        else:
            canvas = np.full((base_h, base_w, 4), 255, dtype=np.uint8)
            canvas[:, :, 0:3] = self.bg_color

        canvas = self.overlay_pixel(canvas, face_img, 0, 0)
        p = OFFSETS.get(pose, OFFSETS["front"])

        # Mata
        l_mode = state["eyes_left"]; l_asset = self.assets["eyes_left"].get(l_mode)
        if l_asset is not None: canvas = self.transform_and_overlay(canvas, l_asset, p["left_eye"])

        r_mode = state["eyes_right"]; r_asset = self.assets["eyes_right"].get(r_mode)
        if r_asset is not None: canvas = self.transform_and_overlay(canvas, r_asset, p["right_eye"])

        # Mulut
        mouth_mode = state["mouth"]; mouth = self.assets["mouth"].get(mouth_mode)
        if mouth is not None: 
            mx, my, msx, msy, mr = p["mouth"]
            tx, ty = MOUTH_TWEAKS.get(mouth_mode, (0, 0))
            canvas = self.transform_and_overlay(canvas, mouth, [mx + tx, my + ty, msx, msy, mr])

        return canvas

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    def calculate_ear(self, lm, idx):
        p = [np.array([lm[i].x, lm[i].y]) for i in idx]
        v1 = np.linalg.norm(p[1] - p[5]); v2 = np.linalg.norm(p[2] - p[4]); h = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * h)

    def calculate_mar(self, lm, idx):
        p = [np.array([lm[i].x, lm[i].y]) for i in idx]
        return np.linalg.norm(p[2] - p[3]) / np.linalg.norm(p[0] - p[1])

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        status = {
            "pose": "front", 
            "eyes_left": "normal", 
            "eyes_right": "normal", 
            "mouth": "idle"
        }
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            nose_ratio = (lm[1].x - lm[234].x) / (lm[454].x - lm[234].x)
            if nose_ratio < 0.35: status["pose"] = "left" 
            elif nose_ratio > 0.65: status["pose"] = "right"  
            else:
                diff_y = lm[1].y - (lm[159].y + lm[386].y) / 2
                if diff_y < 0.03: status["pose"] = "up"
                elif diff_y > 0.12: status["pose"] = "down"
                else: status["pose"] = "front"

            l_ear = self.calculate_ear(lm, [33, 160, 158, 133, 153, 144])
            r_ear = self.calculate_ear(lm, [362, 385, 387, 263, 373, 380])
            if l_ear < 0.20: status["eyes_left"] = "blink"
            elif l_ear < 0.28: status["eyes_left"] = "squint"
            else: status["eyes_left"] = "normal"

            if r_ear < 0.20: status["eyes_right"] = "blink"
            elif r_ear < 0.28: status["eyes_right"] = "squint"
            else: status["eyes_right"] = "normal"

            mar = self.calculate_mar(lm, [61, 291, 13, 14])
            if mar > 0.3: status["mouth"] = "open"
            elif mar > 0.05: status["mouth"] = "talk"
            else: status["mouth"] = "idle"

        return status

def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()
    renderer = VTuberRenderer()
    
    # Fitur Resizable Window
    win_name = "VTuber"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720) # Default HD

    print(">> VTUBER FINAL (RESIZABLE) <<")
    print(">> 'O' : OPEN Control Panel")
    print(">> 'X' : EXIT Control Panel")
    print(">> 'S' : SAVE Config")
    print(">> 'Q' : QUIT Program")

    last_pose = "none"
    show_panel = False 

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        state = tracker.detect(frame)
        pose = state["pose"]

        if show_panel:
            if cv2.getWindowProperty("Control Panel", cv2.WND_PROP_VISIBLE) < 1:
                cv2.namedWindow("Control Panel")
                cv2.resizeWindow("Control Panel", 400, 750)
                for part in ["L Eye", "R Eye", "Mouth"]:
                    cv2.createTrackbar(f"{part} X", "Control Panel", 0, 2000, nothing)
                    cv2.createTrackbar(f"{part} Y", "Control Panel", 0, 2000, nothing)
                    cv2.createTrackbar(f"{part} Rot", "Control Panel", 180, 360, nothing) 
                    cv2.createTrackbar(f"{part} Scale X", "Control Panel", 100, 200, nothing)
                    cv2.createTrackbar(f"{part} Scale Y", "Control Panel", 100, 200, nothing)
                last_pose = "force_redraw"

            if pose != last_pose:
                p = OFFSETS[pose]
                cv2.setTrackbarPos("L Eye X", "Control Panel", int(p["left_eye"][0]))
                cv2.setTrackbarPos("L Eye Y", "Control Panel", int(p["left_eye"][1]))
                cv2.setTrackbarPos("L Eye Scale X", "Control Panel", int(p["left_eye"][2]))
                cv2.setTrackbarPos("L Eye Scale Y", "Control Panel", int(p["left_eye"][3]))
                cv2.setTrackbarPos("L Eye Rot", "Control Panel", int(p["left_eye"][4]) + 180)
                
                cv2.setTrackbarPos("R Eye X", "Control Panel", int(p["right_eye"][0]))
                cv2.setTrackbarPos("R Eye Y", "Control Panel", int(p["right_eye"][1]))
                cv2.setTrackbarPos("R Eye Scale X", "Control Panel", int(p["right_eye"][2]))
                cv2.setTrackbarPos("R Eye Scale Y", "Control Panel", int(p["right_eye"][3]))
                cv2.setTrackbarPos("R Eye Rot", "Control Panel", int(p["right_eye"][4]) + 180)

                cv2.setTrackbarPos("Mouth X", "Control Panel", int(p["mouth"][0]))
                cv2.setTrackbarPos("Mouth Y", "Control Panel", int(p["mouth"][1]))
                cv2.setTrackbarPos("Mouth Scale X", "Control Panel", int(p["mouth"][2]))
                cv2.setTrackbarPos("Mouth Scale Y", "Control Panel", int(p["mouth"][3]))
                cv2.setTrackbarPos("Mouth Rot", "Control Panel", int(p["mouth"][4]) + 180)
                last_pose = pose
            else:
                lx = cv2.getTrackbarPos("L Eye X", "Control Panel"); ly = cv2.getTrackbarPos("L Eye Y", "Control Panel")
                lsx = cv2.getTrackbarPos("L Eye Scale X", "Control Panel"); lsy = cv2.getTrackbarPos("L Eye Scale Y", "Control Panel")
                lr = cv2.getTrackbarPos("L Eye Rot", "Control Panel") - 180
                OFFSETS[pose]["left_eye"] = [lx, ly, lsx, lsy, lr]

                rx = cv2.getTrackbarPos("R Eye X", "Control Panel"); ry = cv2.getTrackbarPos("R Eye Y", "Control Panel")
                rsx = cv2.getTrackbarPos("R Eye Scale X", "Control Panel"); rsy = cv2.getTrackbarPos("R Eye Scale Y", "Control Panel")
                rr = cv2.getTrackbarPos("R Eye Rot", "Control Panel") - 180
                OFFSETS[pose]["right_eye"] = [rx, ry, rsx, rsy, rr]

                mx = cv2.getTrackbarPos("Mouth X", "Control Panel"); my = cv2.getTrackbarPos("Mouth Y", "Control Panel")
                msx = cv2.getTrackbarPos("Mouth Scale X", "Control Panel"); msy = cv2.getTrackbarPos("Mouth Scale Y", "Control Panel")
                mr = cv2.getTrackbarPos("Mouth Rot", "Control Panel") - 180
                OFFSETS[pose]["mouth"] = [mx, my, msx, msy, mr]

            panel_img = np.zeros((50, 400, 3), dtype=np.uint8)
            cv2.putText(panel_img, f"EDIT: {pose.upper()} | [S] SAVE", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Control Panel", panel_img)
        else:
            try: cv2.destroyWindow("Control Panel")
            except: pass

        vtuber_view = renderer.render(state)
        
        # PROSES RESIZE PINTAR
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
            if win_w <= 0: win_w, win_h = 1280, 720
        except: win_w, win_h = 1280, 720
            
        final_display = resize_keep_aspect_ratio(vtuber_view, win_w, win_h)
        cv2.imshow(win_name, final_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'): save_config(OFFSETS)
        elif key == ord('o'): show_panel = True; last_pose = "force_update"
        elif key == ord('x'): show_panel = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()