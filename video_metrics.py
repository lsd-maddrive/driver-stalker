import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

LEFT_EYE_INDICES = [33, 160, 159, 158, 133, 153, 145, 144] 
RIGHT_EYE_INDICES = [362, 385, 386, 387, 263, 373, 374, 380]
MOUTH_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 
                       185, 40, 39, 37, 0, 267, 269, 270, 409, 375, 321, 405, 314, 17, 84, 181, 91, 146]
FACE_POSE_INDICES = [1, 152, 33, 263, 61, 291]
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],        
    [0.0, -330.0, -65.0],    
    [-225.0, 170.0, -135.0], 
    [225.0, 170.0, -135.0],  
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]  
], dtype=np.float64)

def detect_eye(data):
    A = distance.euclidean(data[2], data[6])
    B = distance.euclidean(data[3], data[5])
    D = distance.euclidean(data[0], data[4])
    return (A + B) / (2 * D)

def detect_mouth(data):
    A = distance.euclidean(data[24], data[33])
    C = distance.euclidean(data[5], data[15])
    D = distance.euclidean(data[6], data[14])
    width = distance.euclidean(data[0], data[10])
    return (A + C + D) / (3 * width)

def get_head_pose(landmarks, img_size):
    points_2d = np.array([(landmarks[i].x * img_size[1], landmarks[i].y * img_size[0]) for i in FACE_POSE_INDICES], dtype=np.float64)
    
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    _, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, points_2d, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

eyes_closed_start_time = None
def process_frame(frame):
    global eyes_closed_start_time
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    state = "Normal condition"
    ear, mar = 0, 0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]
            landmarks = face_landmarks.landmark
            
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_INDICES]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_INDICES]
            mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_INNER_INDICES]

            ear = (detect_eye(left_eye) + detect_eye(right_eye)) / 2
            mar = detect_mouth(mouth)
            pitch, yaw, roll = get_head_pose(landmarks, (h, w))

            current_time = time.time()
            if ear < 0.24 and mar < 0.3: 
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = current_time
                else:
                    if current_time - eyes_closed_start_time >= 0.5:
                        state = "SLEEP"
            else:
                eyes_closed_start_time = None
                state = "Normal condition"
            
            if mar > 0.33:
                state = "Yawning"
            
            face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
            face_oval_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i, _ in face_oval]
            face_oval_points = np.array(face_oval_points, dtype=np.int32)
            cv2.polylines(frame, [face_oval_points], True, (255, 255, 255), 1)


            for eye_points in [left_eye, right_eye]:
                for i in range(len(eye_points)):
                    start = eye_points[i]
                    end = eye_points[(i + 1) % len(eye_points)]
                    color = (255, 255, 0) if eye_points == right_eye else (255, 255, 0)
                    cv2.line(frame, start, end, color, 2)
        
            for i in range(len(mouth)):
                start = mouth[i]
                end = mouth[(i + 1) % len(mouth)]
                cv2.line(frame, start, end, (0, 128, 255), 2)
    
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {state}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, state, ear, mar

input_path = r"C:\\Users\\ACER\\Documents\\VKR\\Video\\P1043115_na1.mp4"  
output_path = r"C:\\Users\\ACER\\Documents\\VKR\\Video\\result1.mp4"  

cap = cv2.VideoCapture(input_path)
    
if output_path:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
while cap.isOpened():
    ret, frame = cap.read()      
    processed_frame, state, ear, mar = process_frame(frame)
        
    if output_path:
        out.write(processed_frame)   
    cv2.imshow('Face State Detection', processed_frame)   
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()


