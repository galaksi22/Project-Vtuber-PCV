import cv2, mediapipe as mp, numpy as np
import os, json, math, traceback, subprocess, sys

# ================= CONFIGURATION =================
ASSET_DIR, BG_FILENAME = "assets", "Background.png"
WINDOW_NAME, EXTERNAL_SCRIPT = "VTuber Final Engine", "pose_tracking.py"
BODY_CONFIG_FILE, FACE_CONFIG_FILE = "vtuber_body.json", "vtuber_face.json"

# Physics & Animation
BREATH_SPEED, BREATH_AMPLITUDE = 0.08, 3
LEAN_SENSITIVITY, SMOOTH_FACTOR, TRANSITION_FRAMES = 10.0, 0.12, 15

# Image Processing (Sharpen Kernel)
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Assets Map
BODY_ASSETS_MAP = {
    "NORMAL": "badan_full.png",
    "BOTH_UP": "dua_tangan_naik.png", "BOTH_T": "dua_tangan_T.png",
    "BOTH_1": "dua_tangan_1.png", "BOTH_2": "dua_tangan_2.png", "BOTH_3": "dua_tangan_3.png", "BOTH_4": "dua_tangan_4.png", "BOTH_5": "dua_tangan_5.png",
    "RIGHT_UP": "tangan_kanan_naik.png", "RIGHT_PEACE": "tangan_kanan_peace.png", "RIGHT_THUMB": "tangan_kanan_jempol.png",
    "RIGHT_1": "tangan_kanan_1.png", "RIGHT_2": "tangan_kanan_2.png", "RIGHT_3": "tangan_kanan_3.png", "RIGHT_4": "tangan_kanan_4.png", "RIGHT_5": "tangan_kanan_5.png",
    "LEFT_UP": "tangan_kiri_naik.png", "LEFT_PEACE": "tangan_kiri_peace.png", "LEFT_THUMB": "tangan_kiri_jempol.png",
    "LEFT_1": "tangan_kiri_1.png", "LEFT_2": "tangan_kiri_2.png", "LEFT_3": "tangan_kiri_3.png", "LEFT_4": "tangan_kiri_4.png", "LEFT_5": "tangan_kiri_5.png",
}

# Default Transforms
DEFAULT_OFFSETS = {
    "global_transform": {"x": 640, "y": 700, "body_scale": 60, "head_scale": 15, "total_scale": 100, "bg_index": 0},
    "front": { "neck": [0, -50], "head_transform": [0, 100, 100], "left_eye": [800, 950, 100, 100, 0], "right_eye": [1200, 950, 100, 100, 0], "mouth": [1000, 1150, 100, 100, 0] }
}
for p in ["left", "right", "up", "down"]:
    if p not in DEFAULT_OFFSETS: DEFAULT_OFFSETS[p] = DEFAULT_OFFSETS["front"].copy()

# ================= UTILITIES & MATH =================
def ensure_list_length(lst, target_len, fill_value):
    if not isinstance(lst, list): return [fill_value] * target_len
    return lst[:target_len] if len(lst) >= target_len else lst + [fill_value] * (target_len - len(lst))

def validate_data(pose_data):
    pose_data["neck"] = ensure_list_length(pose_data.get("neck", [0, -50]), 2, 0)
    pose_data["head_transform"] = ensure_list_length(pose_data.get("head_transform", [0, 100, 100]), 3, 100)
    if pose_data["head_transform"][0] == 100: pose_data["head_transform"][0] = 0
    
    for feat in ["left_eye", "right_eye", "mouth"]:
        if feat not in pose_data: pose_data[feat] = DEFAULT_OFFSETS["front"][feat]
        curr = pose_data[feat]
        if len(curr) < 5:
            pose_data[feat] = [curr[0], curr[1], curr[2], curr[2], 0] if len(curr) == 3 else ensure_list_length(curr, 5, 0)
        if pose_data[feat][2] <= 0: pose_data[feat][2] = 100
        if pose_data[feat][3] <= 0: pose_data[feat][3] = 100
    return pose_data

def load_config():
    combined = DEFAULT_OFFSETS.copy()
    try:
        if os.path.exists(BODY_CONFIG_FILE):
            with open(BODY_CONFIG_FILE, 'r') as f:
                d = json.load(f)
                if "global_transform" in d: combined["global_transform"].update(d["global_transform"])
                for p in ["front", "left", "right", "up", "down"]:
                    if p in d and "neck" in d[p]: combined[p]["neck"] = d[p]["neck"]
        if os.path.exists(FACE_CONFIG_FILE):
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
    body["global_transform"] = data["global_transform"]
    for p in ["front", "left", "right", "up", "down"]:
        body[p] = { "neck": data[p]["neck"] }
        face[p] = { k: data[p][k] for k in ["head_transform", "left_eye", "right_eye", "mouth"] }
    try:
        with open(BODY_CONFIG_FILE, 'w') as f: json.dump(body, f, indent=4)
        with open(FACE_CONFIG_FILE, 'w') as f: json.dump(face, f, indent=4)
        print("[SAVED] Config updated.")
    except Exception as e: print(f"[ERROR SAVE]: {e}")

def nothing(x): pass
def calc_dist(p1, p2): return math.hypot(p1.x - p2.x, p1.y - p2.y)
def calculate_angle(p1, p2): return math.degrees(math.atan2(p2.x - p1.x, p1.y - p2.y))

def calculate_3_point_angle(a, b, c):
    ba, bc = np.array([a.x - b.x, a.y - b.y]), np.array([c.x - b.x, c.y - b.y])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def rotate_point(p, center, angle_deg):
    rad = math.radians(angle_deg)
    xr, yr = p.x - center.x, p.y - center.y
    return (xr * math.sin(rad) + yr * math.cos(rad)) + center.y

def rotate_image_safe(image, angle):
    if image is None: return None, None
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nw, nh = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - (w / 2); M[1, 2] += (nh / 2) - (h / 2)
    return cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)), M

OFFSETS = load_config()

# ================= TRACKING LOGIC =================
def get_arm_pose(pose_lm):
    lm = pose_lm.landmark; status = {"left": "DOWN", "right": "DOWN"}
    tilt_deg = math.degrees(math.atan2(lm[11].y - lm[12].y, lm[11].x - lm[12].x))
    
    # Sensitivity & Rotation Check
    SENSITIVITY = 0.005
    if rotate_point(lm[16], lm[12], -tilt_deg) < rotate_point(lm[0], lm[12], -tilt_deg) - SENSITIVITY: status["right"] = "UP"
    if rotate_point(lm[15], lm[11], -tilt_deg) < rotate_point(lm[0], lm[11], -tilt_deg) - SENSITIVITY: status["left"] = "UP"

    # T-Pose Check
    if (abs(lm[15].y - lm[11].y) < 0.15 and calculate_3_point_angle(lm[11], lm[13], lm[15]) > 150) and \
       (abs(lm[16].y - lm[12].y) < 0.15 and calculate_3_point_angle(lm[12], lm[14], lm[16]) > 150):
        status["left"] = status["right"] = "T-POSE"
    
    status['tilt'] = tilt_deg
    return status

class HybridTracker:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(refine_face_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_finger_array(self, hand_lm):
        lm = hand_lm.landmark; f = []
        f.append(1 if calc_dist(lm[4], lm[17]) > calc_dist(lm[3], lm[17]) else 0) # Thumb
        for t, p in zip([8, 12, 16, 20], [6, 10, 14, 18]): f.append(1 if calc_dist(lm[t], lm[0]) > calc_dist(lm[p], lm[0]) else 0)
        return f

    def get_detailed_gesture(self, fingers, hand_lm, pose_lm):
        lm = hand_lm.landmark
        if calc_dist(lm[4], lm[8]) < 0.05 and fingers[2:] == [1,1,1]: return "OK"
        if fingers in [[0,1,1,0,0], [1,1,1,0,0]]:
            is_chest = lm[0].y > pose_lm.landmark[0].y if pose_lm else True
            return "PEACE" if abs(calculate_angle(lm[0], lm[8])) > 20 and is_chest else "2"
        if abs(calculate_angle(lm[0], lm[9])) < 30: # Upright check
            if fingers == [1,1,1,1,1]: return "5"
            if fingers == [0,1,1,1,1]: return "4"
            if fingers in [[0,1,1,1,0], [1,1,1,1,0]]: return "3"
            if fingers in [[0,1,0,0,0], [1,1,0,0,0]]: return "1"
        return "THUMB" if fingers == [1,0,0,0,0] else "NONE"

    def get_eye_ratio(self, lm, pts):
        h = calc_dist(lm[pts[2]], lm[pts[3]])
        return calc_dist(lm[pts[0]], lm[pts[1]]) / h if h > 0 else 0

    def detect(self, img):
        res = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        st = {"pose": "front", "eyes_left": "normal", "eyes_right": "normal", "mouth": "idle", "gesture": "NORMAL", "tilt": 0, "nose_x": 0.5}
        l_gest, r_gest, arm_l, arm_r = "NONE", "NONE", "DOWN", "DOWN"

        if res.pose_landmarks:
            p_stat = get_arm_pose(res.pose_landmarks)
            arm_l, arm_r, st["tilt"] = p_stat["left"], p_stat["right"], p_stat["tilt"]
            st["nose_x"] = res.pose_landmarks.landmark[0].x

        if res.left_hand_landmarks: l_gest = self.get_detailed_gesture(self.get_finger_array(res.left_hand_landmarks), res.left_hand_landmarks, res.pose_landmarks)
        if res.right_hand_landmarks: r_gest = self.get_detailed_gesture(self.get_finger_array(res.right_hand_landmarks), res.right_hand_landmarks, res.pose_landmarks)

        # Gesture Logic
        if arm_l == "T-POSE" or arm_r == "T-POSE": st["gesture"] = "BOTH_T"
        elif arm_l == "UP" and arm_r == "UP": st["gesture"] = "BOTH_UP"
        elif arm_l == "UP": st["gesture"] = "RIGHT_UP"
        elif arm_r == "UP": st["gesture"] = "LEFT_UP"
        elif l_gest == r_gest and l_gest in ["1","2","3","4","5"]: st["gesture"] = f"BOTH_{l_gest}"
        else:
            final = "NORMAL"
            if r_gest in ["PEACE", "THUMB", "1", "2", "3", "4", "5"]: final = f"LEFT_{r_gest}"
            if final == "NORMAL" and l_gest in ["PEACE", "THUMB", "1", "2", "3", "4", "5"]: final = f"RIGHT_{l_gest}"
            st["gesture"] = final

        if res.face_landmarks:
            lm = res.face_landmarks.landmark
            r = (lm[1].x - lm[234].x) / (lm[454].x - lm[234].x)
            st["pose"] = "left" if r < 0.35 else ("right" if r > 0.65 else ("up" if (lm[1].y - (lm[159].y + lm[386].y)/2) < 0.03 else ("down" if (lm[1].y - (lm[159].y + lm[386].y)/2) > 0.12 else "front")))
            
            el_ratio = self.get_eye_ratio(lm, [159, 145, 33, 133])
            er_ratio = self.get_eye_ratio(lm, [386, 374, 362, 263])
            st["eyes_left"] = "blink" if el_ratio < 0.16 else ("wide" if el_ratio > 0.38 else "normal")
            st["eyes_right"] = "blink" if er_ratio < 0.16 else ("wide" if er_ratio > 0.38 else "normal")
            
            # Mouth Logic
            ref = calc_dist(lm[33], lm[263])
            if ref > 0:
                ver, hor = calc_dist(lm[13], lm[14]) / ref, calc_dist(lm[61], lm[291]) / ref
                m_val = "4" if hor > 0.60 else ("1" if ver < 0.06 else ("2" if ver < 0.30 else "3"))
                st["mouth"] = {"2": "talk", "3": "open", "4": "laugh"}.get(m_val, "idle")
                st["debug"] = {"mouth_val": m_val}
        
        return st

# ================= RENDERER =================
class VTuberRenderer:
    def __init__(self):
        self.bg_color = (0, 255, 0)
        self.bg_files = [f for f in os.listdir(ASSET_DIR) if ('background' in f.lower() or 'bg' in f.lower()) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if BG_FILENAME not in self.bg_files and os.path.exists(os.path.join(ASSET_DIR, BG_FILENAME)): self.bg_files.insert(0, BG_FILENAME)
        
        self.bg_index = OFFSETS["global_transform"].get("bg_index", 0) % (len(self.bg_files) or 1)
        self.bg_image, self.bg_image_next, self.transition_alpha = None, None, 1.0
        self.load_assets()

    def load_img(self, filename):
        path = os.path.join(ASSET_DIR, filename)
        if not os.path.exists(path): return np.zeros((10, 10, 4), dtype=np.uint8)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return np.zeros((10, 10, 4), dtype=np.uint8)
        if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def start_bg_transition(self, direction=1):
        if not self.bg_files: return
        self.bg_index = (self.bg_index + direction) % len(self.bg_files)
        self.bg_image_next = self.load_img(self.bg_files[self.bg_index])
        self.transition_alpha = 0.99
        OFFSETS["global_transform"]["bg_index"] = self.bg_index
        print(f"[BG] Transition to: {self.bg_files[self.bg_index]}")

    def load_assets(self):
        if self.bg_files: self.bg_image = self.load_img(self.bg_files[self.bg_index])
        self.assets = {
            "base": { k: self.load_img(v) for k,v in {"front":"Muka_Depan.png", "left":"Muka_Kiri.png", "right":"Muka_Kanan.png", "up":"Muka_Atas.png", "down":"muka_bawah.png"}.items() },
            "eyes_left": { k: self.load_img(v) for k,v in {"normal":"Mata_Kanan_2.png", "wide":"Mata_Kanan_3.png", "blink":"Mata_Kanan_1.png"}.items() },
            "eyes_right": { k: self.load_img(v) for k,v in {"normal":"Mata_Kiri_2.png", "wide":"Mata_Kiri_3.png", "blink":"Mata_Kiri_1.png"}.items() },
            "mouth": { k: self.load_img(v) for k,v in {"idle":"Mulut_1.png", "talk":"Mulut_2.png", "open":"Mulut_3.png", "laugh":"Mulut_4.png"}.items() },
            "bodies": { k: self.load_img(v) for k,v in BODY_ASSETS_MAP.items() }
        }

    def smart_resize_bg(self, bg, target_w, target_h):
        if bg is None:
            c = np.full((target_h, target_w, 4), 255, dtype=np.uint8); c[:,:,:3] = self.bg_color; return c
        h, w = bg.shape[:2]; scale = max(target_w/w, target_h/h)
        resized = cv2.resize(bg, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        x, y = max(0, (resized.shape[1]-target_w)//2), max(0, (resized.shape[0]-target_h)//2)
        crop = resized[y:y+target_h, x:x+target_w]
        return cv2.resize(crop, (target_w, target_h)) if crop.shape[:2] != (target_h, target_w) else crop

    def get_current_bg(self, w, h):
        cur = self.smart_resize_bg(self.bg_image, w, h)
        if self.transition_alpha < 1.0 and self.bg_image_next is not None:
            nxt = self.smart_resize_bg(self.bg_image_next, w, h)
            return cv2.addWeighted(nxt, 1.0 - self.transition_alpha, cur, self.transition_alpha, 0)
        return cur

    def overlay(self, bg, fg, x, y):
        hf, wf = fg.shape[:2]; hb, wb = bg.shape[:2]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(wb, x+wf), min(hb, y+hf)
        if x1 >= x2 or y1 >= y2: return bg
        fg_c = fg[y1-y:y2-y, x1-x:x2-x]
        if fg_c.shape[2] == 4:
            alpha = fg_c[:,:,3:4] / 255.0
            bg[y1:y2, x1:x2, :3] = (fg_c[:,:,:3] * alpha) + (bg[y1:y2, x1:x2, :3] * (1.0 - alpha))
        else: bg[y1:y2, x1:x2] = fg_c
        return bg

    def render(self, state, win_w, win_h, lean_angle=0, breath_offset=0):
        canvas = self.get_current_bg(win_w, win_h)
        glob = OFFSETS["global_transform"]
        win_scale = min(win_w/1280.0, win_h/720.0)
        tot_sc = glob.get("total_scale", 100) / 100.0
        
        # Calculate Scales & Anchor
        b_sc, h_sc = (max(5, glob["body_scale"])/100.0)*win_scale*tot_sc, (max(5, glob["head_scale"])/100.0)*win_scale*tot_sc
        ax, ay = int(win_w * (glob["x"]/1280.0)), int(win_h * (glob["y"]/720.0))
        
        # 1. Body Render
        body = self.assets["bodies"].get(state.get("gesture", "NORMAL"), self.assets["bodies"]["NORMAL"])
        bh, bw = body.shape[:2]; nbw, nbh = int(bw*b_sc), int(bh*b_sc)
        body_rot, M_body = rotate_image_safe(cv2.resize(body, (nbw, nbh), interpolation=cv2.INTER_AREA), lean_angle)
        
        feet = M_body.dot(np.array([nbw/2, nbh, 1]))
        bx, by = int(ax - feet[0]), int(ay - feet[1]) + int(breath_offset)
        canvas = self.overlay(canvas, body_rot, bx, by)

        # 2. Head Setup
        p_data = OFFSETS.get(state["pose"], OFFSETS["front"])
        h_trans = p_data.get("head_transform", [0, 100, 100])
        tot_head_rot = h_trans[0] + lean_angle
        hsx, hsy = max(10, h_trans[1])/100.0, max(10, h_trans[2])/100.0
        
        head_raw = self.assets["base"].get(state["pose"], self.assets["base"]["front"])
        hh, hw = head_raw.shape[:2]; chw, chh = int(hw*h_sc*hsx), int(hh*h_sc*hsy)
        head_rot, M_head = rotate_image_safe(cv2.resize(head_raw, (chw, chh), interpolation=cv2.INTER_AREA), tot_head_rot)
        
        # Neck Calculation
        neck = p_data.get("neck", [0, -50])
        neck_pos = M_body.dot(np.array([(nbw/2)+int(neck[0]*win_scale*tot_sc), int(neck[1]*win_scale*tot_sc), 1]))
        hx, hy = int((bx + neck_pos[0]) - M_head.dot(np.array([chw/2, chh, 1]))[0]), int((by + neck_pos[1]) - M_head.dot(np.array([chw/2, chh, 1]))[1])
        canvas = self.overlay(canvas, head_rot, hx, hy)

        # 3. Features Render
        def draw_feat(key, param):
            raw = self.assets["eyes_left" if "left" in param else ("eyes_right" if "right" in param else "mouth")].get(key)
            if raw is None: return
            ox, oy, osx, osy, orot = p_data[param]
            fsx, fsy = (osx/100.0)*h_sc*hsx, (osy/100.0)*h_sc*hsy
            
            ex_x, ex_y = (-20, -100) if (param == "mouth" and key == "laugh") else (0, 0)
            if ex_x: fsx *= 0.25; fsy *= 0.25
            
            nfh, nfw = int(raw.shape[0]*fsy), int(raw.shape[1]*fsx)
            f_rot, _ = rotate_image_safe(cv2.resize(raw, (nfw, nfh), interpolation=cv2.INTER_AREA), orot + tot_head_rot)
            
            feat_pos = M_head.dot(np.array([(chw/2)+((ox-1000+ex_x)*h_sc*hsx), (chh/2)+((oy-1000+ex_y)*h_sc*hsy), 1]))
            return self.overlay(canvas, f_rot, int(hx + feat_pos[0] - f_rot.shape[1]/2), int(hy + feat_pos[1] - f_rot.shape[0]/2))

        canvas = draw_feat(state["eyes_left"], "left_eye")
        canvas = draw_feat(state["eyes_right"], "right_eye")
        return draw_feat(state["mouth"], "mouth")

# ================= UI & APP LOOP =================
def draw_ui_panel(img, x, y, w, h, transparency=0.6):
    ovr = img.copy()
    c_top, c_bot, c_bor = np.array([60, 40, 30]), np.array([20, 10, 5]), (255, 200, 0, 255)
    grad = np.tile(np.linspace(c_top, c_bot, h).astype(np.uint8)[:, np.newaxis, :], (1, w, 1))
    if img.shape[2] == 4: grad = cv2.cvtColor(grad, cv2.COLOR_BGR2BGRA)
    ovr[y:y+h, x:x+w] = grad
    cv2.addWeighted(ovr, transparency, img, 1-transparency, 0, img)
    
    r, th = 15, 2
    cv2.line(img, (x+r, y), (x+w-r, y), c_bor, th); cv2.line(img, (x+r, y+h), (x+w-r, y+h), c_bor, th)
    cv2.line(img, (x, y+r), (x, y+h-r), c_bor, th); cv2.line(img, (x+w, y+r), (x+w, y+h-r), c_bor, th)
    for c, ang in [((x+r, y+r), 180), ((x+w-r, y+r), 270), ((x+w-r, y+h-r), 0), ((x+r, y+h-r), 90)]:
        cv2.ellipse(img, c, (r, r), ang, 0, 90, c_bor, th)

def run_vtuber_app_loop(cap):
    tr, ren = HybridTracker(), VTuberRenderer()
    show_ovr = show_glob = glob_init = ovr_init = show_debug = False
    curr_lean, frame_cnt = 0, 0
    
    # Flags & Indices
    is_sharp, is_breath, is_lean = False, True, True
    last_p, last_part, curr_part_idx = "none", -1, 0
    PARTS_KEY = ["left_eye", "right_eye", "mouth", "neck", "head_transform"]
    PARTS_NAME = ["Mata Kiri", "Mata Kanan", "Mulut", "Leher", "Kepala"]
    
    # Colors
    C_TITLE, C_LBL, C_VAL = (255, 255, 255), (230, 220, 100), (180, 210, 240)
    C_KEY, C_DESC, C_ON, C_OFF = (255, 200, 100), (255, 255, 255), (255, 200, 0), (80, 80, 80)

    def update_tb(win, name, val):
        try: cv2.setTrackbarPos(name, win, cv2.getTrackbarPos(name, win) + val)
        except: pass

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1
        st = tr.detect(cv2.flip(frame, 1))
        
        # Physics Update
        lean_target = st.get("tilt", 0) * (LEAN_SENSITIVITY/-90.0)
        curr_lean += (max(-40, min(40, lean_target)) - curr_lean) * SMOOTH_FACTOR if is_lean else -curr_lean * SMOOTH_FACTOR
        breath_y = math.sin(frame_cnt * BREATH_SPEED) * BREATH_AMPLITUDE if is_breath else 0

        # BG Transition
        if ren.transition_alpha < 1.0:
            ren.transition_alpha -= (1.0/TRANSITION_FRAMES)
            if ren.transition_alpha <= 0.0:
                ren.bg_image = ren.bg_image_next; ren.bg_image_next = None; ren.transition_alpha = 1.0; save_config(OFFSETS)

        # Controls
        key = cv2.waitKeyEx(1)
        if key != -1:
            ck = key & 0xFF
            if ck == 27: return "EXIT"
            elif ck == ord('d'): return "SWITCH"
            elif ck == ord('s'): save_config(OFFSETS)
            elif ck in [ord('h'), ord('H')]: show_debug = not show_debug
            elif ck in [ord('k'), ord('K')]: is_sharp = not is_sharp
            elif ck in [ord('l'), ord('L')]: is_lean = not is_lean
            elif ck in [ord('z'), ord('Z')]: is_breath = not is_breath
            elif ck in [ord('o'), ord('O')]: show_ovr = not show_ovr; show_glob = False
            elif ck in [ord('p'), ord('P')]: show_glob = not show_glob; show_ovr = False
            elif ck in [ord('x'), ord('X')]: show_ovr = show_glob = False
            elif ren.bg_image_next is None and ck in [ord('.'), ord('>'), ord(','), ord('<')]:
                ren.start_bg_transition(1 if ck in [ord('.'), ord('>')] else -1)
            
            # Editor Hotkeys
            if show_ovr:
                if key == 2424832: update_tb("EDITOR", "POS X / ROT", -1); update_tb("EDITOR", "ROTASI", -1) # Left
                elif key == 2555904: update_tb("EDITOR", "POS X / ROT", 1); update_tb("EDITOR", "ROTASI", 1) # Right
                elif key == 2490368: update_tb("EDITOR", "POS Y / S_X", -1); update_tb("EDITOR", "SCALE X", 1); update_tb("EDITOR", "SCALE Y", 1) # Up
                elif key == 2621440: update_tb("EDITOR", "POS Y / S_X", 1); update_tb("EDITOR", "SCALE X", -1); update_tb("EDITOR", "SCALE Y", -1) # Down
            if show_glob:
                if key == 2424832: update_tb("GLOBAL", "Pos X", -1); update_tb("GLOBAL", "Body Sc", -1); update_tb("GLOBAL", "Head Sc", -1)
                elif key == 2555904: update_tb("GLOBAL", "Pos X", 1); update_tb("GLOBAL", "Body Sc", 1); update_tb("GLOBAL", "Head Sc", 1)
                elif key == 2490368: update_tb("GLOBAL", "Pos Y", -1); update_tb("GLOBAL", "Total Sc", 1)
                elif key == 2621440: update_tb("GLOBAL", "Pos Y", 1); update_tb("GLOBAL", "Total Sc", -1)

        # Logic Window Editor (Global)
        if show_glob:
            if not glob_init:
                cv2.namedWindow("GLOBAL"); cv2.resizeWindow("GLOBAL", 400, 300); glob_init = True
                g = OFFSETS["global_transform"]
                for n, v, m in [("Pos X", g["x"], 1280), ("Pos Y", g["y"], 2000), ("Body Sc", g["body_scale"], 200), ("Head Sc", g["head_scale"], 100), ("Total Sc", g.get("total_scale", 100), 300)]:
                    cv2.createTrackbar(n, "GLOBAL", int(v), m, nothing)
            try:
                g = OFFSETS["global_transform"]
                g["x"], g["y"] = cv2.getTrackbarPos("Pos X", "GLOBAL"), cv2.getTrackbarPos("Pos Y", "GLOBAL")
                g["body_scale"], g["head_scale"], g["total_scale"] = cv2.getTrackbarPos("Body Sc", "GLOBAL"), cv2.getTrackbarPos("Head Sc", "GLOBAL"), cv2.getTrackbarPos("Total Sc", "GLOBAL")
            except: pass
        elif glob_init: cv2.destroyWindow("GLOBAL"); glob_init = False

        # Logic Window Editor (Parts)
        if show_ovr:
            if not ovr_init:
                cv2.namedWindow("EDITOR"); cv2.resizeWindow("EDITOR", 400, 350); ovr_init = True; last_p = "force"
                for n, v, m in [("PILIH BAGIAN", 0, 4), ("POS X / ROT", 1000, 2000), ("POS Y / S_X", 1000, 2000), ("SCALE X", 100, 200), ("SCALE Y", 100, 200), ("ROTASI", 180, 360)]:
                    cv2.createTrackbar(n, "EDITOR", v, m, nothing)
            try:
                curr_part_idx = cv2.getTrackbarPos("PILIH BAGIAN", "EDITOR")
                curr_key, p_data = PARTS_KEY[curr_part_idx], OFFSETS.get(st["pose"], OFFSETS["front"])
                
                if curr_part_idx != last_part or st["pose"] != last_p:
                    v = p_data[curr_key]
                    if curr_part_idx == 3: # Neck
                        cv2.setTrackbarPos("POS X / ROT", "EDITOR", int(v[0]+500)); cv2.setTrackbarPos("POS Y / S_X", "EDITOR", int(v[1]+500))
                    elif curr_part_idx == 4: # Head
                        cv2.setTrackbarPos("POS X / ROT", "EDITOR", int(v[0]+180)); cv2.setTrackbarPos("POS Y / S_X", "EDITOR", int(v[1]))
                        cv2.setTrackbarPos("SCALE X", "EDITOR", int(v[2])); cv2.setTrackbarPos("SCALE Y", "EDITOR", int(v[3])); cv2.setTrackbarPos("ROTASI", "EDITOR", int(v[4]+180))
                    else: # Features
                        cv2.setTrackbarPos("POS X / ROT", "EDITOR", int(v[0])); cv2.setTrackbarPos("POS Y / S_X", "EDITOR", int(v[1]))
                        cv2.setTrackbarPos("SCALE X", "EDITOR", int(v[2])); cv2.setTrackbarPos("SCALE Y", "EDITOR", int(v[3])); cv2.setTrackbarPos("ROTASI", "EDITOR", int(v[4]+180))
                    last_part, last_p = curr_part_idx, st["pose"]
                else:
                    vals = [cv2.getTrackbarPos(t, "EDITOR") for t in ["POS X / ROT", "POS Y / S_X", "SCALE X", "SCALE Y", "ROTASI"]]
                    if curr_part_idx == 3: OFFSETS[st["pose"]]["neck"] = [vals[0]-500, vals[1]-500]
                    elif curr_part_idx == 4: OFFSETS[st["pose"]]["head_transform"] = [vals[0]-180, vals[1], vals[2]]
                    else: OFFSETS[st["pose"]][curr_key] = [vals[0], vals[1], vals[2], vals[3], vals[4]-180]
            except: pass
        elif ovr_init: cv2.destroyWindow("EDITOR"); ovr_init = False

        # Rendering & Post-Process
        try: _, _, ww, wh = cv2.getWindowImageRect(WINDOW_NAME)
        except: ww, wh = 1280, 720
        view = ren.render(st, max(1, ww), max(1, wh), curr_lean, breath_y)
        if is_sharp: view = cv2.filter2D(view, -1, SHARPEN_KERNEL)

        # UI Overlay
        FONT = cv2.FONT_HERSHEY_DUPLEX
        if show_debug:
            draw_ui_panel(view, 20, 20, 440, 365, 0.6)
            sy, lx, rx = 50, 40, 210
            
            cv2.putText(view, "== MENU NAVIGASI & STATUS ==", (lx, sy), FONT, 0.5, C_TITLE, 1, cv2.LINE_AA); sy+=30
            cv2.putText(view, "BACKGROUND:", (lx, sy), FONT, 0.5, C_LBL, 1, cv2.LINE_AA)
            cv2.putText(view, f"{ren.bg_files[ren.bg_index] if ren.bg_files else 'None'}", (rx, sy), FONT, 0.5, C_VAL, 1, cv2.LINE_AA); sy+=25
            cv2.putText(view, "GESTURE :", (lx, sy), FONT, 0.5, C_LBL, 1, cv2.LINE_AA)
            cv2.putText(view, f"{st['gesture']}", (rx, sy), FONT, 0.5, C_VAL, 1, cv2.LINE_AA); sy+=25
            
            cv2.putText(view, "EFFECTS   :", (lx, sy), FONT, 0.5, C_LBL, 1, cv2.LINE_AA)
            for name, stat, px in [("Sway (L)", is_lean, rx), ("Breath (Z)", is_breath, rx+110)]:
                col = C_ON if stat else C_OFF
                cv2.circle(view, (px, sy-5), 5, col, -1)
                if stat: cv2.circle(view, (px, sy-5), 8, col, 1)
                cv2.putText(view, name, (px+15, sy), FONT, 0.5, C_VAL, 1, cv2.LINE_AA)
            sy+=25
            col = C_ON if is_sharp else C_OFF
            cv2.circle(view, (rx, sy-5), 5, col, -1)
            if is_sharp: cv2.circle(view, (rx, sy-5), 8, col, 1)
            cv2.putText(view, "Sharp (K)", (rx+15, sy), FONT, 0.5, C_VAL, 1, cv2.LINE_AA); sy+=35

            shortcuts = [("[ESC]", "Keluar"), ("[K]", "Efek Tajam"), ("[P]", "Global Transform"), ("[O]", "Editor Bagian"), ("[S]", "Simpan Config"), ("[X]", "Tutup Menu"), ("[< / >]", "Ganti BG"), ("[H]", "Sembunyikan Menu")]
            for k, d in shortcuts:
                cv2.putText(view, k, (lx, sy), FONT, 0.5, C_KEY, 1, cv2.LINE_AA)
                cv2.putText(view, d, (lx+110, sy), FONT, 0.5, C_DESC, 1, cv2.LINE_AA); sy+=25
        else:
            cv2.putText(view, "Tekan [H] untuk Bantuan/Menu", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if show_ovr:
            txt_p, txt_e = f"POSE MODE: {st['pose'].upper()}", f"EDITING  : {PARTS_NAME[curr_part_idx]}"
            t1, _ = cv2.getTextSize(txt_p, FONT, 0.6, 1); t2, _ = cv2.getTextSize(txt_e, FONT, 0.6, 1)
            cv2.putText(view, txt_p, (ww-t1[0]-40, 50), FONT, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(view, txt_e, (ww-t2[0]-40, 80), FONT, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, view)

# ================= ENTRY POINT =================
if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        print(">> SYSTEM READY <<")
        mode = "VTUBER"
        
        while True:
            if mode == "VTUBER": res = run_vtuber_app_loop(cap)
            else:
                cap.release(); cv2.destroyAllWindows()
                try: subprocess.run([sys.executable, EXTERNAL_SCRIPT])
                except Exception as e: print(f"ERROR {EXTERNAL_SCRIPT}: {e}")
                cap = cv2.VideoCapture(0); cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.resizeWindow(WINDOW_NAME, 1280, 720); mode = "VTUBER"; continue
            
            if res == "SWITCH": mode = "SKELETON" if mode == "VTUBER" else "VTUBER"
            elif res == "EXIT": break
            
        cap.release(); cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n!!! CRASH: {e}"); traceback.print_exc(); input("Enter to exit...")