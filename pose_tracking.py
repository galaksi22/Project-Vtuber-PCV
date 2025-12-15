import cv2
import mediapipe as mp
import numpy as np
import math

# ================= KONFIGURASI =================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# --- FUNGSI BARU UNTUK MEROTASI TITIK ---
def rotate_point(p, center, angle_deg):
    """Memutar titik landmark (p) di sekitar titik pusat (center) sebesar angle_deg."""
    angle_rad = math.radians(angle_deg)
    
    # Koordinat X dan Y MediaPipe adalah ternormalisasi (0 hingga 1)
    
    # Translasi ke asal (center menjadi (0,0))
    x_rel = p.x - center.x
    y_rel = p.y - center.y
    
    # Rotasi
    x_rot = x_rel * math.cos(angle_rad) - y_rel * math.sin(angle_rad)
    y_rot = x_rel * math.sin(angle_rad) + y_rel * math.cos(angle_rad)
    
    # Translasi kembali dan kembalikan hanya Y yang diputar
    return y_rot + center.y 

# --- FUNGSI MATEMATIKA ---
def calc_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_angle(p1, p2):
    dx = p2.x - p1.x; dy = p2.y - p1.y
    return math.degrees(math.atan2(dx, -dy))

def calculate_3_point_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def calculate_body_tilt(pose_lm):
    lm = pose_lm.landmark
    p1 = lm[11] # Bahu Kiri
    p2 = lm[12] # Bahu Kanan
    
    angle_rad = math.atan2(p1.y - p2.y, p1.x - p2.x)
    return math.degrees(angle_rad) 

# --- FUNGSI LOGIKA MULUT (HORIZONTAL & VERTIKAL) (TIDAK BERUBAH) ---
def get_mouth_status(face_lm):
    lm = face_lm.landmark
    v_dist = calc_dist(lm[13], lm[14])
    h_dist = calc_dist(lm[61], lm[291])
    ref_width = calc_dist(lm[33], lm[263])
    if ref_width == 0: return "1", 0, 0 
    ratio_vertical = v_dist / ref_width
    ratio_horizontal = h_dist / ref_width
    THRESH_SMILE_WIDTH = 0.61
    THRESH_DIAM = 0.06
    THRESH_MED = 0.20
    if ratio_horizontal > THRESH_SMILE_WIDTH:
        return "4", ratio_vertical, ratio_horizontal
    if ratio_vertical < THRESH_DIAM:
        return "1", ratio_vertical, ratio_horizontal # Diam
    elif ratio_vertical < THRESH_MED:
        return "2", ratio_vertical, ratio_horizontal # Bicara Biasa
    else:
        return "3", ratio_vertical, ratio_horizontal # Buka Lebar (A/O)

# --- FUNGSI JARI (TIDAK BERUBAH) ---
def get_finger_array(hand_lm):
    lm = hand_lm.landmark
    fingers = []
    if calc_dist(lm[4], lm[17]) > calc_dist(lm[3], lm[17]): fingers.append(1)
    else: fingers.append(0)
    tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if calc_dist(lm[tip], lm[0]) > calc_dist(lm[pip], lm[0]): fingers.append(1)
        else: fingers.append(0)
    return fingers

def get_detailed_gesture(fingers, hand_lm, pose_lm):
    lm = hand_lm.landmark
    gesture = "NONE"
    tilt_deg = abs(calculate_angle(lm[0], lm[9]))
    is_upright = tilt_deg < 30 
    
    if calc_dist(lm[4], lm[8]) < 0.05 and fingers[2:] == [1,1,1]: return "OK \U0001F44C"
    
    if (fingers[1]==1 and fingers[2]==1 and fingers[3]==0 and fingers[4]==0):
        idx_tilt = abs(calculate_angle(lm[0], lm[8]))
        is_chest_height = True 
        if pose_lm: is_chest_height = lm[0].y > pose_lm.landmark[0].y 
        if idx_tilt > 20 and is_chest_height: gesture = "PEACE \u270C\uFE0F (Miring)"
        else: gesture = "ANGKA 2 2️⃣"
    elif fingers == [1, 1, 1, 1, 1]: 
        if is_upright: gesture = "OPEN (5) \u270B"
    elif fingers == [0, 1, 1, 1, 1]: 
        if is_upright: gesture = "EMPAT (4)"
    elif fingers == [0, 1, 1, 1, 0] or fingers == [1, 1, 1, 1, 0]: 
        if is_upright: gesture = "TIGA (3)"
    elif fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]: 
        if is_upright: gesture = "SATU (1) \u261D\uFE0F"
    elif fingers == [1, 0, 0, 0, 0]: gesture = "JEMPOL \U0001F44D"
    elif fingers == [0, 0, 0, 0, 0]: gesture = "FIST (0) \u270A"
    elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]: gesture = "ROCK \U0001F918"
    elif fingers == [1, 1, 0, 0, 1]: gesture = "SPIDER-MAN \U0001F91F"
    return gesture

# ... (Semua fungsi di atas tetap sama) ...

# --- FUNGSI get_arm_pose (LOGIKA UP BERDASARKAN GARIS MERAH - FIXED SIMETRI) ---
def get_arm_pose(pose_lm):
    lm = pose_lm.landmark
    status = {"left": "DOWN", "right": "DOWN"}
    
    tilt_deg = calculate_body_tilt(pose_lm) 
    
    ls, le, lw = lm[11], lm[13], lm[15] # Left Shoulder, Elbow, Wrist
    rs, re, rw = lm[12], lm[14], lm[16] # Right Shoulder, Elbow, Wrist
    nose = lm[0]                       # Hidung (Landmark 0)
    
    # **********************************************
    # 1. Rotasi Titik Kanan (Menggunakan Bahu Kanan sebagai Pusat)
    # **********************************************
    rotated_rw_y = rotate_point(rw, rs, -tilt_deg)
    # Ambang Batas Kanan (Hidung dirotasi di sekitar Bahu Kanan)
    rotated_nose_r_y = rotate_point(nose, rs, -tilt_deg)
    is_wrist_high_r = rotated_rw_y < rotated_nose_r_y
    
    # **********************************************
    # 2. Rotasi Titik Kiri (Menggunakan Bahu Kiri sebagai Pusat)
    # **********************************************
    rotated_lw_y = rotate_point(lw, ls, -tilt_deg)
    # Ambang Batas Kiri (Hidung dirotasi di sekitar Bahu Kiri)
    rotated_nose_l_y = rotate_point(nose, ls, -tilt_deg)
    is_wrist_high_l = rotated_lw_y < rotated_nose_l_y
    
    # Logika T-Pose (TIDAK BERUBAH)
    angle_l = calculate_3_point_angle(ls, le, lw)
    angle_r = calculate_3_point_angle(rs, re, rw)
    raw_t_left = abs(lw.y - ls.y) < 0.15 and abs(le.y - ls.y) < 0.15 and angle_l > 150
    raw_t_right = abs(rw.y - rs.y) < 0.15 and abs(re.y - rs.y) < 0.15 and angle_r > 150
    
    if is_wrist_high_l: status["left"] = "UP \u2B06\uFE0F (Relatif)"
    if is_wrist_high_r: status["right"] = "UP \u2B06\uFE0F (Relatif)"
    
    if raw_t_left and raw_t_right:
        status["left"] = "T-POSE \u2708\uFE0F"
        status["right"] = "T-POSE \u2708\uFE0F"
    
    status['tilt'] = tilt_deg
    return status

# --- (SISA KODE main() DI BAWAH INI TETAP SAMA SEPERTI SEBELUMNYA) ---

def main():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            h, w, _ = image.shape

            # --- DETEKSI MULUT ---
            mouth_val = "1"; ver_deb = 0; hor_deb = 0
            if results.face_landmarks:
                mouth_val, ver_deb, hor_deb = get_mouth_status(results.face_landmarks)
                lm = results.face_landmarks.landmark
                for p in [33, 263, 13, 14, 61, 291]:
                    cx, cy = int(lm[p].x * w), int(lm[p].y * h)
                    cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
                p1 = (int(lm[33].x * w), int(lm[33].y * h)); p2 = (int(lm[263].x * w), int(lm[263].y * h))
                cv2.line(image, p1, p2, (255,0,0), 1)

            # --- DETEKSI BADAN ---
            pose_landmarks = results.pose_landmarks
            arm_status = {"left": "-", "right": "-", "tilt": 0}
            if pose_landmarks:
                arm_status = get_arm_pose(pose_landmarks)
                mp_drawing.draw_landmarks(image, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
                lm = pose_landmarks.landmark
                tilt_deg = arm_status['tilt']
                angle_rad = math.radians(tilt_deg)
                M = math.tan(angle_rad)
                
                # TITIK LANDMARK KUNCI DALAM PIXEL (untuk visualisasi)
                shoulder_x = int((lm[11].x + lm[12].x) / 2 * w); shoulder_y = int((lm[11].y + lm[12].y) / 2 * h)
                nose_x = int(lm[0].x * w); nose_y = int(lm[0].y * h)
                
                # VISUALISASI GARIS MIRING SEJAJAR
                
                # 1. GARIS KUNING (CYAN) - Melewati Titik Bahu
                C_yellow = shoulder_y - M * shoulder_x
                y1_yellow = int(C_yellow); y2_yellow = int(M * w + C_yellow)
                cv2.line(image, (0, y1_yellow), (w, y2_yellow), (0, 255, 255), 1)

                # 2. GARIS MERAH - Melewati Titik Hidung
                C_red = nose_y - M * nose_x
                y1_red = int(C_red); y2_red = int(M * w + C_red)
                cv2.line(image, (0, y1_red), (w, y2_red), (0, 0, 255), 1)
                
            # --- DETEKSI TANGAN ---
            l_gesture = "-"
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if "UP" not in arm_status['left']:
                    l_fingers = get_finger_array(results.left_hand_landmarks)
                    l_gesture = get_detailed_gesture(l_fingers, results.left_hand_landmarks, pose_landmarks)
                wrist = results.left_hand_landmarks.landmark[0]
                col = (255, 255, 255) if l_gesture == "-" else (0, 255, 255)
                cv2.putText(image, l_gesture, (int(wrist.x*w), int(wrist.y*h)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            r_gesture = "-"
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if "UP" not in arm_status['right']:
                    r_fingers = get_finger_array(results.right_hand_landmarks)
                    r_gesture = get_detailed_gesture(r_fingers, results.right_hand_landmarks, pose_landmarks)
                wrist = results.right_hand_landmarks.landmark[0]
                col = (255, 255, 255) if r_gesture == "-" else (0, 255, 255)
                cv2.putText(image, r_gesture, (int(wrist.x*w), int(wrist.y*h)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            # --- DISPLAY INFO ---
            cv2.rectangle(image, (0,0), (640, 120), (0,0,0), -1)
            cv2.putText(image, f"L: {l_gesture} | Arm: {arm_status['left']} | Tilt: {arm_status['tilt']:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, f"R: {r_gesture} | Arm: {arm_status['right']}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # Logic Teks Mulut
            desc = "DIAM"
            if mouth_val == "2": desc = "BICARA (Kecil)"
            elif mouth_val == "3": desc = "BUKA (Lebar)"
            elif mouth_val == "4": desc = "MERINGIS/TAWA (Gigi)"
            
            cv2.putText(image, f"MOUTH: {mouth_val} ({desc})", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(image, f"DEBUG -> Ver: {ver_deb:.3f} | Hor: {hor_deb:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow('Smart Tracking V16 (Final Relative UP Fixed)', image)
            if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()