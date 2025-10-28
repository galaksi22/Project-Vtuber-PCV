import cv2 as cv
import mediapipe as mp

# Setup MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# Buka webcam
cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Biar nggak mirror kebalik
    frame = cv.flip(frame, 1)

    # Convert ke RGB
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Kalau ada wajah terdeteksi
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,  # garis full mesh
                mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=1)
            )

    cv.imshow("Face Mesh", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
