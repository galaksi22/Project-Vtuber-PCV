import cv2
import mediapipe as mp
import numpy as np
import os
import json
import math
import traceback
import subprocess
import sys

# ================= KONFIGURASI UMUM =================
ASSET_DIR = "assets"
BG_FILENAME = "Background.png" 
WINDOW_NAME = "VTuber Final Engine"
EXTERNAL_SCRIPT = "pose_tracking.py"

# --- CONFIG ANIMASI ---
BREATH_SPEED = 0.08 
BREATH_AMPLITUDE = 3 
LEAN_SENSITIVITY = 10.0 
SMOOTH_FACTOR = 0.12 
TRANSITION_FRAMES = 15 

# --- CONFIG FILES ---
BODY_CONFIG_FILE = "vtuber_body.json"
FACE_CONFIG_FILE = "vtuber_face.json"

# ==============================================================================
# BAGIAN 1: LOGIKA VTUBER
# ==============================================================================

BODY_ASSETS_MAP = {
    "NORMAL": "badan_full.png",
    "BOTH_UP": "dua_tangan_naik.png", "BOTH_T": "dua_tangan_T.png",
    "BOTH_1": "dua_tangan_1.png", "BOTH_2": "dua_tangan_2.png", "BOTH_3": "dua_tangan_3.png", "BOTH_4": "dua_tangan_4.png", "BOTH_5": "dua_tangan_5.png",
    "RIGHT_UP": "tangan_kanan_naik.png", "RIGHT_PEACE": "tangan_kanan_peace.png", "RIGHT_THUMB": "tangan_kanan_jempol.png",
    "RIGHT_1": "tangan_kanan_1.png", "RIGHT_2": "tangan_kanan_2.png", "RIGHT_3": "tangan_kanan_3.png", "RIGHT_4": "tangan_kanan_4.png", "RIGHT_5": "tangan_kanan_5.png",
    "LEFT_UP": "tangan_kiri_naik.png", "LEFT_PEACE": "tangan_kiri_peace.png", "LEFT_THUMB": "tangan_kiri_jempol.png",
    "LEFT_1": "tangan_kiri_1.png", "LEFT_2": "tangan_kiri_2.png", "LEFT_3": "tangan_kiri_3.png", "LEFT_4": "tangan_kiri_4.png", "LEFT_5": "tangan_kiri_5.png",
}

DEFAULT_OFFSETS = {
    "global_transform": {"x": 640, "y": 700, "body_scale": 60, "head_scale": 15, "total_scale": 100, "bg_index": 0},
    "front": { "neck": [0, -50], "head_transform": [0, 100, 100], "left_eye": [800, 950, 100, 100, 0], "right_eye": [1200, 950, 100, 100, 0], "mouth": [1000, 1150, 100, 100, 0] }
}
for p in ["left", "right", "up", "down"]:
    if p not in DEFAULT_OFFSETS: DEFAULT_OFFSETS[p] = DEFAULT_OFFSETS["front"].copy()

# --- UTILITIES ---
def ensure_list_length(lst, target_len, fill_value):
    if not isinstance(lst, list): return [fill_value] * target_len
    if len(lst) >= target_len: return lst[:target_len]
    return lst + [fill_value] * (target_len - len(lst))

def validate_data(pose_data):
    if "neck" not in pose_data: pose_data["neck"] = [0, -50]
    pose_data["neck"] = ensure_list_length(pose_data["neck"], 2, 0)
    if "head_transform" not in pose_data: pose_data["head_transform"] = [0, 100, 100]
    pose_data["head_transform"] = ensure_list_length(pose_data["head_transform"], 3, 100)
    if len(pose_data["head_transform"]) == 3 and pose_data["head_transform"][0] == 100: pose_data["head_transform"][0] = 0
    for feat in ["left_eye", "right_eye", "mouth"]:
        if feat not in pose_data: pose_data[feat] = DEFAULT_OFFSETS["front"][feat]
        curr = pose_data[feat]
        if len(curr) < 5:
            if len(curr) == 3: pose_data[feat] = [curr[0], curr[1], curr[2], curr[2], 0]
            elif len(curr) == 4: pose_data[feat] = [curr[0], curr[1], curr[2], curr[2], curr[3]]
            else: pose_data[feat] = ensure_list_length(curr, 5, 0)
            if pose_data[feat][2] <= 0: pose_data[feat][2] = 100
            if pose_data[feat][3] <= 0: pose_data[feat][3] = 100
    return pose_data

def load_config():
    combined = DEFAULT_OFFSETS.copy()
    if os.path.exists(BODY_CONFIG_FILE):
        try:
            with open(BODY_CONFIG_FILE, 'r') as f:
                d = json.load(f)
                if "global_transform" in d: 
                    combined["global_transform"] = d["global_transform"]
                    if "total_scale" not in combined["global_transform"]:
                        combined["global_transform"]["total_scale"] = 100
                    if "bg_index" not in combined["global_transform"]:
                        combined["global_transform"]["bg_index"] = 0
                for p in ["front", "left", "right", "up", "down"]:
                    if p in d and "neck" in d[p]: combined[p]["neck"] = d[p]["neck"]
        except: pass
    if os.path.exists(FACE_CONFIG_FILE):
        try:
            with open(FACE_CONFIG_FILE, 'r') as f:
                d = json.load(f)
                for p in ["front", "left", "right", "up", "down"]:
                    if p in d:
                        if "head_transform" in d[p]: combined[p]["head_transform"] = d[p]["head_transform"]
                        for feat in ["left_eye", "right_eye", "mouth"]:
                            if feat in d[p]: combined[p][feat] = d[p][feat]
        except: pass
    for p in ["front", "left", "right", "up", "down"]: combined[p] = validate_data(combined[p])
    return combined

def save_config(data):
    body, face = {}, {}
    body["global_transform"] = {
        "x": data["global_transform"]["x"],
        "y": data["global_transform"]["y"],
        "body_scale": data["global_transform"]["body_scale"],
        "head_scale": data["global_transform"]["head_scale"],
        "total_scale": data["global_transform"]["total_scale"],
        "bg_index": data["global_transform"].get("bg_index", 0) 
    }
    for p in ["front", "left", "right", "up", "down"]:
        body[p] = { "neck": data[p]["neck"] }
        face[p] = { "head_transform": data[p]["head_transform"], "left_eye": data[p]["left_eye"], "right_eye": data[p]["right_eye"], "mouth": data[p]["mouth"] }
    try:
        with open(BODY_CONFIG_FILE, 'w') as f: json.dump(body, f, indent=4)
        with open(FACE_CONFIG_FILE, 'w') as f: json.dump(face, f, indent=4)
        print("[SAVED] Config updated.")
    except Exception as e: print(f"[ERROR SAVE]: {e}")

def nothing(x): pass

def calc_dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)
def calculate_angle(p1, p2): return math.degrees(math.atan2(p2.x - p1.x, p1.y - p2.y))
def calculate_3_point_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y]); bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_body_tilt(pose_lm):
    lm = pose_lm.landmark
    p1 = lm[11] # Bahu Kiri
    p2 = lm[12] # Bahu Kanan
    angle_rad = math.atan2(p1.y - p2.y, p1.x - p2.x)
    return math.degrees(angle_rad) 

def rotate_point(p, center, angle_deg):
    angle_rad = math.radians(angle_deg)
    x_rel = p.x - center.x
    y_rel = p.y - center.y
    x_rot = x_rel * math.cos(angle_rad) - y_rel * math.sin(angle_rad)
    y_rot = x_rel * math.sin(angle_rad) + y_rel * math.cos(angle_rad)
    return y_rot + center.y 

def rotate_image_safe(image, angle):
    if image is None: return None, None
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - (w / 2)
    M[1, 2] += (nh / 2) - (h / 2)
    rotated = cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated, M

OFFSETS = load_config()

# --- FUNGSI get_arm_pose ---
def get_arm_pose(pose_lm):
    lm = pose_lm.landmark
    status = {"left": "DOWN", "right": "DOWN"}
    
    tilt_deg = calculate_body_tilt(pose_lm) 
    
    ls, le, lw = lm[11], lm[13], lm[15] # Left Shoulder, Elbow, Wrist
    rs, re, rw = lm[12], lm[14], lm[16] # Right Shoulder, Elbow, Wrist
    nose = lm[0] # Titik 0: Hidung 
    
    OFFSET_UP_SENSITIVITY = 0.005 
    
    rotated_threshold_r_y = rotate_point(nose, rs, -tilt_deg) - OFFSET_UP_SENSITIVITY 
    rotated_rw_y = rotate_point(rw, rs, -tilt_deg)
    is_wrist_high_r = rotated_rw_y < rotated_threshold_r_y 

    rotated_threshold_l_y = rotate_point(nose, ls, -tilt_deg) - OFFSET_UP_SENSITIVITY 
    rotated_lw_y = rotate_point(lw, ls, -tilt_deg)
    is_wrist_high_l = rotated_lw_y < rotated_threshold_l_y
    
    angle_l = calculate_3_point_angle(ls, le, lw)
    angle_r = calculate_3_point_angle(rs, re, rw)
    raw_t_left = abs(lw.y - ls.y) < 0.15 and abs(le.y - ls.y) < 0.15 and angle_l > 150
    raw_t_right = abs(rw.y - rs.y) < 0.15 and abs(re.y - rs.y) < 0.15 and angle_r > 150
    
    if is_wrist_high_l: status["left"] = "UP"
    if is_wrist_high_r: status["right"] = "UP"
    
    if raw_t_left and raw_t_right:
        status["left"] = "T-POSE"
        status["right"] = "T-POSE"
    
    status['tilt'] = tilt_deg
    return status


# --- CLASS RENDERER VTUBER ---
class VTuberRenderer:
    def __init__(self):
        self.assets = {}; 
        self.bg_color = (0, 255, 0); 
        self.bg_image = None; 
        
        self.bg_files = self.get_bg_files()
        
        saved_index = OFFSETS["global_transform"].get("bg_index", 0)
        num_files = len(self.bg_files)
        self.bg_index = saved_index % num_files if num_files > 0 else 0
        
        self.bg_image_next = None 
        self.transition_alpha = 1.0 
        self.load_assets()

    def get_bg_files(self):
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        files = []
        for f in os.listdir(ASSET_DIR):
            if ('background' in f.lower() or 'bg' in f.lower()):
                for ext in supported_formats:
                    if f.lower().endswith(ext):
                        files.append(f)
                        break
        
        if BG_FILENAME not in files and os.path.exists(os.path.join(ASSET_DIR, BG_FILENAME)):
            files.insert(0, BG_FILENAME)
            
        return files

    def load_img(self, filename):
        path = os.path.join(ASSET_DIR, filename)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
            if img is not None: 
                if img.ndim == 3 and img.shape[2] == 3:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                elif img.ndim == 3 and img.shape[2] == 4:
                    return img
                elif img.ndim == 2:
                     return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        return np.zeros((10, 10, 4), dtype=np.uint8)

    def start_bg_transition(self, direction=1): 
        if not self.bg_files: 
            print("[BG SHIFT FAILED] Tidak ada file background yang terdeteksi.")
            return 
        
        num_files = len(self.bg_files)
        next_index = (self.bg_index + direction + num_files) % num_files
        next_filename = self.bg_files[next_index]
        
        temp_next_img = cv2.imread(os.path.join(ASSET_DIR, next_filename))
        
        if temp_next_img is not None:
            if temp_next_img.ndim == 3 and temp_next_img.shape[2] == 3:
                self.bg_image_next = cv2.cvtColor(temp_next_img, cv2.COLOR_BGR2BGRA)
            elif temp_next_img.ndim == 2:
                self.bg_image_next = cv2.cvtColor(temp_next_img, cv2.COLOR_GRAY2BGRA)
            else: 
                self.bg_image_next = temp_next_img
                
            self.bg_index = next_index
            self.transition_alpha = 0.99 
            print(f"[BG SHIFT] Mulai transisi ke: {self.bg_files[self.bg_index]}")
            
            OFFSETS["global_transform"]["bg_index"] = self.bg_index
            
        else:
            self.bg_image_next = None
            print(f"[BG SHIFT FAILED] Gambar BG {next_filename} gagal dimuat.")

    def load_assets(self):
        if self.bg_files:
            bg_filename = self.bg_files[self.bg_index % len(self.bg_files)]
            temp_img = cv2.imread(os.path.join(ASSET_DIR, bg_filename))
            
            if temp_img is not None:
                if temp_img.ndim == 3 and temp_img.shape[2] == 3:
                    self.bg_image = cv2.cvtColor(temp_img, cv2.COLOR_BGR2BGRA)
                elif temp_img.ndim == 2:
                    self.bg_image = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGRA)
                else:
                    self.bg_image = temp_img
            else:
                 self.bg_image = None
        else:
            self.bg_image = None 
            
        self.bg_image_next = None 
        self.transition_alpha = 1.0 

        self.assets = {
            "base": { "front": self.load_img("Muka_Depan.png"), "left": self.load_img("Muka_Kiri.png"), "right": self.load_img("Muka_Kanan.png"), "up": self.load_img("Muka_Atas.png"), "down": self.load_img("muka_bawah.png") },
            "eyes_left": { "normal": self.load_img("Mata_Kanan_2.png"), "wide": self.load_img("Mata_Kanan_3.png"), "blink": self.load_img("Mata_Kanan_1.png") },
            "eyes_right": { "normal": self.load_img("Mata_Kiri_2.png"), "wide": self.load_img("Mata_Kiri_3.png"), "blink": self.load_img("Mata_Kiri_1.png") },
            "mouth": { "idle": self.load_img("Mulut_1.png"), "talk": self.load_img("Mulut_2.png"), "open": self.load_img("Mulut_3.png"), "laugh": self.load_img("Mulut_4.png") },
            "bodies": {}
        }
        for k, v in BODY_ASSETS_MAP.items(): self.assets["bodies"][k] = self.load_img(v)
    
    def smart_resize_bg(self, bg, target_w, target_h):
        if bg is None: 
            canvas = np.full((target_h, target_w, 4), 255, dtype=np.uint8)
            canvas[:, :, 0:3] = self.bg_color # BGR
            return canvas
            
        h, w = bg.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = max(0, (new_w - target_w) // 2)
        y = max(0, (new_h - target_h) // 2)
        cropped = resized[y:y+target_h, x:x+target_w]
        if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
            cropped = cv2.resize(cropped, (target_w, target_h))
            
        if cropped.ndim == 3 and cropped.shape[2] == 3:
            return cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        return cropped

    def get_current_bg(self, win_w, win_h):
        if self.transition_alpha >= 1.0 and self.bg_image_next is None:
            return self.smart_resize_bg(self.bg_image, win_w, win_h)

        current_bg = self.smart_resize_bg(self.bg_image, win_w, win_h)
        next_bg = self.smart_resize_bg(self.bg_image_next, win_w, win_h)

        if current_bg is not None and next_bg is not None and current_bg.shape == next_bg.shape:
             blended = cv2.addWeighted(next_bg, 1.0 - self.transition_alpha, current_bg, self.transition_alpha, 0)
             return blended
        return current_bg

    def overlay(self, bg, fg, x, y):
        h_fg, w_fg = fg.shape[:2]; h_bg, w_bg = bg.shape[:2]
        x1, y1 = max(0, x), max(0, y); x2, y2 = min(w_bg, x + w_fg), min(h_bg, y + h_fg)
        if x1 >= x2 or y1 >= y2: return bg
        fg_crop = fg[y1-y:y2-y, x1-x:x2-x]; bg_crop = bg[y1:y2, x1:x2]
        if fg_crop.shape[2] == 4:
            alpha = np.expand_dims(fg_crop[:, :, 3] / 255.0, axis=2)
            bg[y1:y2, x1:x2, :3] = (fg_crop[:, :, :3] * alpha) + (bg_crop[:, :, :3] * (1.0 - alpha))
        else: bg[y1:y2, x1:x2] = fg_crop
        return bg

    def render(self, state, win_w, win_h, lean_angle=0, breath_offset=0):
        canvas = self.get_current_bg(win_w, win_h)

        glob = OFFSETS["global_transform"]; BASE_W, BASE_H = 1280.0, 720.0
        win_scale = min(win_w / BASE_W, win_h / BASE_H)
        
        total_scale_multiplier = glob.get("total_scale", 100) / 100.0
        b_scale = (max(5, glob["body_scale"]) / 100.0) * win_scale * total_scale_multiplier
        h_scale = (max(5, glob["head_scale"]) / 100.0) * win_scale * total_scale_multiplier
        neck_scale_ratio = win_scale * total_scale_multiplier
        anchor_x = int(win_w * (glob["x"] / BASE_W))
        anchor_y = int(win_h * (glob["y"] / BASE_H)) 
        gesture = state.get("gesture", "NORMAL")
        body_raw = self.assets["bodies"].get(gesture, self.assets["bodies"]["NORMAL"])
        bh, bw = body_raw.shape[:2]; nbw, nbh = int(bw * b_scale), int(bh * b_scale)
        body_res = cv2.resize(body_raw, (nbw, nbh), interpolation=cv2.INTER_AREA)
        body_rot_img, M_body = rotate_image_safe(body_res, lean_angle)
        orig_feet = np.array([nbw/2, nbh, 1])
        new_feet = M_body.dot(orig_feet)
        bx = int(anchor_x - new_feet[0])
        by = int(anchor_y - new_feet[1]) + int(breath_offset)
        canvas = self.overlay(canvas, body_rot_img, bx, by)
        
        pose = state["pose"]; p_data = OFFSETS.get(pose, OFFSETS["front"])
        h_trans = p_data.get("head_transform", [0, 100, 100])
        total_head_rot = h_trans[0] + lean_angle
        h_str_x, h_str_y = max(10, h_trans[1])/100.0, max(10, h_trans[2])/100.0
        head_raw = self.assets["base"].get(pose, self.assets["base"]["front"])
        hh, hw = head_raw.shape[:2]; curr_hw, curr_hh = int(hw * h_scale * h_str_x), int(hh * h_scale * h_str_y)
        head_rot_img, M_head = rotate_image_safe(cv2.resize(head_raw, (curr_hw, curr_hh), interpolation=cv2.INTER_AREA), total_head_rot)
        neck = p_data.get("neck", [0, -50])
        neck_img_x = (nbw / 2) + int(neck[0] * neck_scale_ratio)
        neck_img_y = int(neck[1] * neck_scale_ratio)
        orig_neck = np.array([neck_img_x, neck_img_y, 1])
        new_neck_pos = M_body.dot(orig_neck)
        final_neck_x = bx + new_neck_pos[0]
        final_neck_y = by + new_neck_pos[1]
        head_anchor_orig = np.array([curr_hw/2, curr_hh, 1])
        head_anchor_new = M_head.dot(head_anchor_orig)
        hx = int(final_neck_x - head_anchor_new[0])
        hy = int(final_neck_y - head_anchor_new[1])
        canvas = self.overlay(canvas, head_rot_img, hx, hy)
        
        def draw_feature(img_key, param_key):
            raw = self.assets["eyes_left" if "left" in param_key else ("eyes_right" if "right" in param_key else "mouth")].get(img_key)
            if raw is None: return
            ox, oy, osx, osy, orot = p_data[param_key]
            fsx, fsy = (osx / 100.0) * h_scale * h_str_x, (osy / 100.0) * h_scale * h_str_y
            extra_x, extra_y = 0, 0
            if param_key == "mouth" and img_key == "laugh":
                fsx *= 0.25; fsy *= 0.25; extra_x = -20; extra_y = -100
            fh, fw = raw.shape[:2]; nfw, nfh = int(fw * fsx), int(fh * fsy)
            f_res = cv2.resize(raw, (nfw, nfh), interpolation=cv2.INTER_AREA)
            f_rot, M_feat_local = rotate_image_safe(f_res, orot + total_head_rot)
            feat_rel_x = (ox - 1000 + extra_x) * h_scale * h_str_x
            feat_rel_y = (oy - 1000 + extra_y) * h_scale * h_str_y
            feat_orig_x = (curr_hw / 2) + feat_rel_x
            feat_orig_y = (curr_hh / 2) + feat_rel_y
            feat_vec = np.array([feat_orig_x, feat_orig_y, 1])
            feat_new_pos = M_head.dot(feat_vec)
            fx = int(hx + feat_new_pos[0] - (f_rot.shape[1] / 2))
            fy = int(hy + feat_new_pos[1] - (f_rot.shape[0] / 2))
            return self.overlay(canvas, f_rot, fx, fy)

        canvas = draw_feature(state["eyes_left"], "left_eye")
        canvas = draw_feature(state["eyes_right"], "right_eye")
        return draw_feature(state["mouth"], "mouth")

# --- CLASS TRACKER VTUBER ---
class HybridTracker:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    def get_finger_array(self, hand_lm):
        lm = hand_lm.landmark; fingers = []
        fingers.append(1 if calc_dist(lm[4], lm[17]) > calc_dist(lm[3], lm[17]) else 0)
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            if calc_dist(lm[tip], lm[0]) > calc_dist(lm[pip], lm[0]): fingers.append(1)
            else: fingers.append(0)
        return fingers
    def get_detailed_gesture(self, fingers, hand_lm, pose_lm):
        lm = hand_lm.landmark; is_upright = abs(calculate_angle(lm[0], lm[9])) < 30
        if calc_dist(lm[4], lm[8]) < 0.05 and fingers[2:] == [1,1,1]: return "OK"
        if fingers == [0,1,1,0,0] or fingers == [1,1,1,0,0]:
            is_chest = lm[0].y > pose_lm.landmark[0].y if pose_lm else True
            return "PEACE" if abs(calculate_angle(lm[0], lm[8])) > 20 and is_chest else "2"
        if is_upright:
            if fingers == [1,1,1,1,1]: return "5"
            if fingers == [0,1,1,1,1]: return "4"
            if fingers in [[0,1,1,1,0], [1,1,1,1,0]]: return "3"
            if fingers in [[0,1,0,0,0], [1,1,0,0,0]]: return "1"
        return "THUMB" if fingers == [1,0,0,0,0] else "NONE"
    
    def get_eye_ratio(self, lm, pts):
        v = calc_dist(lm[pts[0]], lm[pts[1]]); h = calc_dist(lm[pts[2]], lm[pts[3]])
        return v / h if h > 0 else 0
        
    def get_mouth_status(self, face_lm):
        lm = face_lm.landmark; ref = calc_dist(lm[33], lm[263])
        if ref == 0: return "1"
        ver, hor = calc_dist(lm[13], lm[14]) / ref, calc_dist(lm[61], lm[291]) / ref
        return "4" if hor > 0.60 else ("1" if ver < 0.06 else ("2" if ver < 0.30 else "3"))
    
    def detect(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); res = self.holistic.process(rgb)
        st = {"pose": "front", "eyes_left": "normal", "eyes_right": "normal", "mouth": "idle", "gesture": "NORMAL", "tilt": 0}
        
        nose_x_norm = 0.5 
        
        arm_l = "DOWN"; arm_r = "DOWN"; l_gest = "NONE"; r_gest = "NONE"; mouth_val = "1"
        
        if res.pose_landmarks:
            pose_status = get_arm_pose(res.pose_landmarks)
            arm_l = pose_status["left"]
            arm_r = pose_status["right"]
            st["tilt"] = pose_status["tilt"] 
            lm = res.pose_landmarks.landmark
            nose_x_norm = lm[0].x 
        
        if res.left_hand_landmarks: l_gest = self.get_detailed_gesture(self.get_finger_array(res.left_hand_landmarks), res.left_hand_landmarks, res.pose_landmarks)
        if res.right_hand_landmarks: r_gest = self.get_detailed_gesture(self.get_finger_array(res.right_hand_landmarks), res.right_hand_landmarks, res.pose_landmarks)
        
        if arm_l == "T-POSE" or arm_r == "T-POSE": 
            st["gesture"] = "BOTH_T"
        elif arm_l == "UP" and arm_r == "UP": 
            st["gesture"] = "BOTH_UP" 
        elif arm_l == "UP": 
            st["gesture"] = "RIGHT_UP" 
        elif arm_r == "UP": 
            st["gesture"] = "LEFT_UP"
        
        elif l_gest == r_gest and l_gest in ["1","2","3","4","5"]: 
            st["gesture"] = f"BOTH_{l_gest}"
        
        else:
            final_pose = "NORMAL"
            if r_gest in ["PEACE", "THUMB", "1", "2", "3", "4", "5"]: final_pose = f"LEFT_{r_gest}"
            if final_pose == "NORMAL" and l_gest in ["PEACE", "THUMB", "1", "2", "3", "4", "5"]: final_pose = f"RIGHT_{l_gest}"
            st["gesture"] = final_pose
        
        if res.face_landmarks:
            lm = res.face_landmarks.landmark; r = (lm[1].x - lm[234].x) / (lm[454].x - lm[234].x)
            st["pose"] = "left" if r < 0.35 else ("right" if r > 0.65 else ("up" if (lm[1].y - (lm[159].y + lm[386].y)/2) < 0.03 else ("down" if (lm[1].y - (lm[159].y + lm[386].y)/2) > 0.12 else "front")))
            st["eyes_left"] = "blink" if self.get_eye_ratio(lm, [159, 145, 33, 133]) < 0.16 else ("wide" if self.get_eye_ratio(lm, [159, 145, 33, 133]) > 0.38 else "normal")
            st["eyes_right"] = "blink" if self.get_eye_ratio(lm, [386, 374, 362, 263]) < 0.16 else ("wide" if self.get_eye_ratio(lm, [386, 374, 362, 263]) > 0.38 else "normal")
            mouth_val = self.get_mouth_status(res.face_landmarks)
            st["mouth"] = {"2": "talk", "3": "open", "4": "laugh"}.get(mouth_val, "idle")
            
        st["debug"] = {"l_gest": l_gest, "r_gest": r_gest, "mouth_val": mouth_val}
        st["nose_x"] = nose_x_norm
        return st

# --- FUNGSI UI MODERN ESTETIK ---
def draw_ui_panel(img, x, y, w, h, transparency=0.6):
    overlay = img.copy()
    COLOR_BG = (45, 35, 20)       # Dark Blue-Black (Elegant)
    COLOR_BORDER = (255, 200, 0)  # Cyan/Emas Cerah (Border)
    r = 15 # Radius sudut
    
    # Isi (Fill) Panel
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), COLOR_BG, -1)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), COLOR_BG, -1)
    cv2.circle(overlay, (x+r, y+r), r, COLOR_BG, -1)
    cv2.circle(overlay, (x+w-r, y+r), r, COLOR_BG, -1)
    cv2.circle(overlay, (x+r, y+h-r), r, COLOR_BG, -1)
    cv2.circle(overlay, (x+w-r, y+h-r), r, COLOR_BG, -1)
    cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)
    
    # Gambar Border (Garis Luar) Tipis
    thick = 1
    cv2.line(img, (x+r, y), (x+w-r, y), COLOR_BORDER, thick)
    cv2.line(img, (x+r, y+h), (x+w-r, y+h), COLOR_BORDER, thick)
    cv2.line(img, (x, y+r), (x, y+h-r), COLOR_BORDER, thick)
    cv2.line(img, (x+w, y+r), (x+w, y+h-r), COLOR_BORDER, thick)
    cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, COLOR_BORDER, thick)
    cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, COLOR_BORDER, thick)
    cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, COLOR_BORDER, thick)
    cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, COLOR_BORDER, thick)

# ==============================================================================
# BAGIAN 3: LOGIKA VTUBER APP 
# ==============================================================================
def run_vtuber_app_loop(cap):
    tr = HybridTracker()
    ren = VTuberRenderer()
    
    show_ovr = False; show_glob = False; glob_init = False
    ovr_init = False; last_p = "none"; last_part = -1; curr_part_idx = 0
    show_debug_ui = False 
    
    PARTS_KEY = ["left_eye", "right_eye", "mouth", "neck", "head_transform"]
    PARTS_NAME = ["Mata Kiri", "Mata Kanan", "Mulut", "Leher", "Kepala"]
    
    curr_lean = 0
    frame_counter = 0
    is_breathing_enabled = True 
    is_leaning_enabled = True    
    
    # --- FONT & WARNA UI BARU ---
    UI_FONT = cv2.FONT_HERSHEY_DUPLEX # Font lebih tegas dan bersih
    C_TITLE = (255, 255, 255) # Putih
    C_LABEL = (230, 220, 100) # Cyan (Mirip referensi)
    C_VAL   = (180, 210, 240) # Emas Pucat (Mirip referensi)
    C_KEY   = (255, 200, 100) # Biru Cyan Terang (Tombol)
    C_DESC  = (255, 255, 255) # Putih (Deskripsi)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1
        state = tr.detect(cv2.flip(frame, 1)); pose = state["pose"]; dbg = state["debug"]
        
        lean_target = state.get("tilt", 0) * (LEAN_SENSITIVITY / -90.0) 
        lean_target = max(-40, min(40, lean_target))
        
        if is_leaning_enabled:
            curr_lean = curr_lean + (lean_target - curr_lean) * SMOOTH_FACTOR
        else:
            curr_lean = 0
            
        if is_breathing_enabled:
            breath_y = math.sin(frame_counter * BREATH_SPEED) * BREATH_AMPLITUDE
        else:
            breath_y = 0
        
        full_key = cv2.waitKeyEx(1)
        
        if ren.transition_alpha < 1.0 and ren.bg_image_next is not None:
            ren.transition_alpha -= (1.0 / TRANSITION_FRAMES) 
            
            if ren.transition_alpha <= 0.0:
                ren.bg_image = ren.bg_image_next
                ren.bg_image_next = None
                ren.transition_alpha = 1.0 
                save_config(OFFSETS)
        
        def update_val(window_name, trackbar_name, direction):
            try:
                pos = cv2.getTrackbarPos(trackbar_name, window_name)
                new_pos = pos + direction
                cv2.setTrackbarPos(trackbar_name, window_name, new_pos)
            except:
                pass
        
        if full_key != -1:
            char_key = full_key & 0xFF
            
            if char_key == 27: return "EXIT"
            elif char_key == ord('d'): return "SWITCH"
            elif char_key == ord('s'): 
                save_config(OFFSETS)
            elif char_key == ord('h'): 
                show_debug_ui = not show_debug_ui
            elif char_key == ord('o'):
                show_ovr = not show_ovr; 
                if show_ovr: show_glob = False
            elif char_key == ord('p'):
                show_glob = not show_glob; 
                if show_glob: show_ovr = False
            elif char_key == ord('x'): 
                show_ovr = False; 
                show_glob = False
            elif char_key == ord('l'): 
                is_leaning_enabled = not is_leaning_enabled
            elif char_key == ord('z'): 
                is_breathing_enabled = not is_breathing_enabled
            
            if ren.bg_image_next is None:
                if char_key == ord('.') or char_key == ord('>'):
                    ren.start_bg_transition(direction=1)
                elif char_key == ord(',') or char_key == ord('<'):
                    ren.start_bg_transition(direction=-1)
            
            if show_ovr:
                if full_key == 2424832:   # PANAH KIRI
                    update_val("EDITOR", "POS X / ROT", -1) 
                    update_val("EDITOR", "ROTASI", -1) 
                elif full_key == 2555904: # PANAH KANAN
                    update_val("EDITOR", "POS X / ROT", 1) 
                    update_val("EDITOR", "ROTASI", 1) 
                elif full_key == 2490368: # PANAH ATAS
                    update_val("EDITOR", "POS Y / S_X", -1) 
                    update_val("EDITOR", "SCALE X", 1) 
                    update_val("EDITOR", "SCALE Y", 1) 
                elif full_key == 2621440: # PANAH BAWAH
                    update_val("EDITOR", "POS Y / S_X", 1) 
                    update_val("EDITOR", "SCALE X", -1) 
                    update_val("EDITOR", "SCALE Y", -1)
            
            if show_glob:
                if full_key == 2424832:   # PANAH KIRI (Pos X -)
                    update_val("GLOBAL", "Pos X", -1)
                    update_val("GLOBAL", "Body Sc", -1) 
                    update_val("GLOBAL", "Head Sc", -1) 
                elif full_key == 2555904: # PANAH KANAN (Pos X +)
                    update_val("GLOBAL", "Pos X", 1)
                    update_val("GLOBAL", "Body Sc", 1) 
                    update_val("GLOBAL", "Head Sc", 1) 
                elif full_key == 2490368: # PANAH ATAS (Pos Y -)
                    update_val("GLOBAL", "Pos Y", -1)
                    update_val("GLOBAL", "Total Sc", 1) 
                elif full_key == 2621440: # PANAH BAWAH (Pos Y +)
                    update_val("GLOBAL", "Pos Y", 1)
                    update_val("GLOBAL", "Total Sc", -1) 


        if show_glob:
            if not glob_init:
                cv2.namedWindow("GLOBAL"); cv2.resizeWindow("GLOBAL", 400, 300); glob_init = True
                g = OFFSETS["global_transform"]
                cv2.createTrackbar("Pos X", "GLOBAL", int(g["x"]), 1280, nothing)
                cv2.createTrackbar("Pos Y", "GLOBAL", int(g["y"]), 2000, nothing) 
                cv2.createTrackbar("Body Sc", "GLOBAL", int(g["body_scale"]), 200, nothing)
                cv2.createTrackbar("Head Sc", "GLOBAL", int(g["head_scale"]), 100, nothing)
                cv2.createTrackbar("Total Sc", "GLOBAL", int(g.get("total_scale", 100)), 300, nothing)
            try:
                g = OFFSETS["global_transform"]
                g["x"] = cv2.getTrackbarPos("Pos X", "GLOBAL")
                g["y"] = cv2.getTrackbarPos("Pos Y", "GLOBAL")
                g["body_scale"] = cv2.getTrackbarPos("Body Sc", "GLOBAL")
                g["head_scale"] = cv2.getTrackbarPos("Head Sc", "GLOBAL")
                g["total_scale"] = cv2.getTrackbarPos("Total Sc", "GLOBAL")
            except: 
                pass
        else:
            if glob_init:
                try: 
                    cv2.destroyWindow("GLOBAL")
                except: 
                    pass
                glob_init = False

        if show_ovr:
            if not ovr_init:
                cv2.namedWindow("EDITOR"); cv2.resizeWindow("EDITOR", 400, 350); ovr_init = True; last_p = "force"; last_part = -1
                cv2.createTrackbar("PILIH BAGIAN", "EDITOR", 0, 4, nothing) 
                cv2.createTrackbar("POS X / ROT", "EDITOR", 1000, 2000, nothing) 
                cv2.createTrackbar("POS Y / S_X", "EDITOR", 1000, 2000, nothing) 
                cv2.createTrackbar("SCALE X", "EDITOR", 100, 200, nothing)
                cv2.createTrackbar("SCALE Y", "EDITOR", 100, 200, nothing)
                cv2.createTrackbar("ROTASI", "EDITOR", 180, 360, nothing)
            try:
                curr_part_idx = cv2.getTrackbarPos("PILIH BAGIAN", "EDITOR")
                curr_key = PARTS_KEY[curr_part_idx]
                p_data = OFFSETS.get(pose, OFFSETS["front"])
                
                if curr_part_idx != last_part or pose != last_p:
                    v = p_data[curr_key]
                    if curr_part_idx == 3: # Neck [x, y]
                        cv2.setTrackbarPos("POS X / ROT", "EDITOR", int(v[0]+500)); 
                        cv2.setTrackbarPos("POS Y / S_X", "EDITOR", int(v[1]+500))
                        cv2.setTrackbarPos("SCALE X", "EDITOR", 100); 
                        cv2.setTrackbarPos("SCALE Y", "EDITOR", 100); 
                        cv2.setTrackbarPos("ROTASI", "EDITOR", 180); 
                    elif curr_part_idx == 4: # Head Transform [rot, sc_x, sc_y]
                        cv2.setTrackbarPos("POS X / ROT", "EDITOR", int(v[0]+180)); 
                        cv2.setTrackbarPos("POS Y / S_X", "EDITOR", int(v[1])); 
                        cv2.setTrackbarPos("SCALE X", "EDITOR", int(v[2])); 
                        cv2.setTrackbarPos("SCALE Y", "EDITOR", int(v[3])); 
                        cv2.setTrackbarPos("ROTASI", "EDITOR", int(v[4]+180)); 
                    else: # Mata/Mulut [ox, oy, sc_x, sc_y, orot]
                        cv2.setTrackbarPos("POS X / ROT", "EDITOR", int(v[0])); 
                        cv2.setTrackbarPos("POS Y / S_X", "EDITOR", int(v[1])); 
                        cv2.setTrackbarPos("SCALE X", "EDITOR", int(v[2])); 
                        cv2.setTrackbarPos("SCALE Y", "EDITOR", int(v[3])); 
                        cv2.setTrackbarPos("ROTASI", "EDITOR", int(v[4]+180))
                    last_part, last_p = curr_part_idx, pose
                else:
                    v = [cv2.getTrackbarPos(t, "EDITOR") for t in ["POS X / ROT", "POS Y / S_X", "SCALE X", "SCALE Y", "ROTASI"]]
                    if curr_part_idx == 3: 
                        OFFSETS[pose]["neck"] = [v[0]-500, v[1]-500]
                    elif curr_part_idx == 4:
                        OFFSETS[pose]["head_transform"] = [v[0]-180, v[1], v[2]]
                    else: 
                        OFFSETS[pose][curr_key] = [v[0], v[1], v[2], v[3], v[4]-180]
            except: 
                pass
        else:
            if ovr_init:
                try: 
                    cv2.destroyWindow("EDITOR")
                except: 
                    pass
                ovr_init = False
        
        try: rect = cv2.getWindowImageRect(WINDOW_NAME); win_w, win_h = (rect[2], rect[3]) if rect and rect[2] > 0 else (1280, 720)
        except: win_w, win_h = 1280, 720
        
        view = ren.render(state, win_w, win_h, lean_angle=curr_lean, breath_offset=breath_y)
        
        # --- UI MODERN & ESTETIK (REPLIKA REFERENSI) ---
        if show_debug_ui:
            draw_ui_panel(view, 20, 20, 480, 380, transparency=0.7)
            
            start_y = 50; left_x = 40; right_x = 210 # Posisi kolom
            
            # 1. JUDUL
            cv2.putText(view, "== MENU NAVIGASI & STATUS ==", (left_x, start_y), UI_FONT, 0.5, C_TITLE, 1, cv2.LINE_AA)
            start_y += 30
            
            # 2. STATUS (Kolom Label & Value)
            # Background
            cv2.putText(view, "BACKGROUND:", (left_x, start_y), UI_FONT, 0.5, C_LABEL, 1, cv2.LINE_AA)
            cv2.putText(view, f"{ren.bg_files[ren.bg_index] if ren.bg_files else 'None'}", (right_x, start_y), UI_FONT, 0.5, C_VAL, 1, cv2.LINE_AA)
            start_y += 25
            # Gesture
            cv2.putText(view, "GESTURE :", (left_x, start_y), UI_FONT, 0.5, C_LABEL, 1, cv2.LINE_AA)
            cv2.putText(view, f"{state['gesture']}", (right_x, start_y), UI_FONT, 0.5, C_VAL, 1, cv2.LINE_AA)
            start_y += 25
            # Sway Toggle
            cv2.putText(view, "SWAY X/Y  :", (left_x, start_y), UI_FONT, 0.5, C_LABEL, 1, cv2.LINE_AA)
            cv2.putText(view, f"{'ON' if is_leaning_enabled or is_breathing_enabled else 'OFF'}", (right_x, start_y), UI_FONT, 0.5, C_VAL, 1, cv2.LINE_AA)
            start_y += 35 # Jarak ekstra sebelum shortcut

            # 3. SHORTCUTS (Kolom Tombol & Deskripsi)
            shortcuts = [
                ("[ESC]", "Keluar Program"),
                ("[P]", "Global Transform (Posisi/Skala)"),
                ("[O]", "Editor Bagian (Mata/Kepala)"),
                ("[S]", "Simpan Konfigurasi"),
                ("[X]", "Tutup Menu Editor/Global"),
                ("[< / >]", "Ganti Background"),
                ("[H]", "Sembunyikan Menu Ini")
            ]
            shortcut_desc_x = 110 # Posisi kolom deskripsi

            for key, desc in shortcuts:
                cv2.putText(view, key, (left_x, start_y), UI_FONT, 0.5, C_KEY, 1, cv2.LINE_AA)
                cv2.putText(view, desc, (left_x + shortcut_desc_x, start_y), UI_FONT, 0.5, C_DESC, 1, cv2.LINE_AA)
                start_y += 25
                
        else:
            cv2.putText(view, "Tekan [H] untuk Bantuan/Menu", (15, 30), UI_FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        if show_ovr: 
            cv2.putText(view, f"POSE: {pose.upper()}", (20,100), UI_FONT, 0.6, (0,0,255), 1)
            cv2.putText(view, f"EDIT: {PARTS_NAME[curr_part_idx]}", (20, 130), UI_FONT, 0.6, (0, 255, 255), 1)
        
        cv2.imshow(WINDOW_NAME, view)

# ==============================================================================
# MAIN CONTROLLER (TIDAK BERUBAH)
# ==============================================================================
def main():
    try:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        print(">> SYSTEM READY <<")
        current_mode = "VTUBER" 
        while True:
            if current_mode == "VTUBER":
                result = run_vtuber_app_loop(cap)
            else:
                cap.release()
                cv2.destroyAllWindows()
                print(">> SWITCHING TO EXTERNAL POSE TRACKING... <<")
                try: subprocess.run([sys.executable, EXTERNAL_SCRIPT])
                except Exception as e: print(f"GAGAL MENJALANKAN {EXTERNAL_SCRIPT}: {e}")
                print(">> RETURNING TO VTUBER... <<")
                cap = cv2.VideoCapture(0)
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 1280, 720)
                current_mode = "VTUBER"
                continue
            if result == "SWITCH": current_mode = "SKELETON" if current_mode == "VTUBER" else "VTUBER"
            elif result == "EXIT": break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("\n!!! PROGRAM CRASH !!!")
        print("Pesan Error:", e)
        traceback.print_exc()
        input("Tekan Enter untuk keluar...")

if __name__ == "__main__":
    main()