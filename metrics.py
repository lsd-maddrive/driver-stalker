import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns

dataset_path = r"C:\\Users\\ACER\\Documents\\VKR\\Dataset\\data\\test"
output_folder = r"C:\\Users\\ACER\\Documents\\test\\results"  
output_csv = os.path.join(output_folder, "results_test.csv")
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

FACE_OUTLINE_INDICES = list(range(0, 468))
LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 469, 470, 471, 472]
RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 474, 475, 476, 477]
MOUTH_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 
                       185, 40, 39, 37, 0, 267, 269, 270, 409, 375, 321, 405, 314, 17, 84, 181, 91, 146]
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],        
    [0.0, -330.0, -65.0],    
    [-225.0, 170.0, -135.0], 
    [225.0, 170.0, -135.0],  
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]  
], dtype=np.float64)

FACE_POSE_INDICES = [1, 152, 33, 263, 61, 291]

def detect_eye(data):
    A = distance.euclidean(data[4], data[12])
    B = distance.euclidean(data[3], data[13])
    C = distance.euclidean(data[0], data[8])
    return (A + B) / (2 * C)

def detect_mouth(data):
    A = distance.euclidean(data[24], data[33])
    C = distance.euclidean(data[5], data[15])
    D = distance.euclidean(data[6], data[14])
    H = distance.euclidean(data[0], data[10])
    return (A + C + D) / (3 * H)

def get_head_pose(landmarks, img_size):
    points_2d = np.array([(landmarks[i].x * img_size[1], landmarks[i].y * img_size[0]) 
                         for i in FACE_POSE_INDICES], dtype=np.float64)
    
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    _, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, points_2d, camera_matrix, 
                                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

def get_true_class(filename):
    filename = filename.lower()
    if 'norm' in filename:
        return 'normal'
    elif 'yawn' in filename:
        return 'yawn'
    elif 'close' in filename:
        return 'close_eyes'

stats = {
    'total_images': len(image_files),
    'faces_detected': 0,
    'no_faces_detected': 0,
    'true_class_dist': defaultdict(int),
    'predicted_dist': defaultdict(int),
    'true_vs_predicted': defaultdict(lambda: defaultdict(int))
}

results = []
valid_results = [] 

for img_file in image_files:
    true_class = get_true_class(img_file)
        
    img_path = os.path.join(dataset_path, img_file)
    img = cv2.imread(img_path)   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(img_rgb)
    
    predicted_state = 'normal'
    ear, mar= 0, 0
    face_detected = False
    
    if results_face.multi_face_landmarks:
        face_detected = True
        stats['faces_detected'] += 1
        
        for face_landmarks in results_face.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _= img.shape
            
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_INDICES]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_INDICES]
            mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH_INNER_INDICES]
            
            ear = (detect_eye(left_eye) + detect_eye(right_eye)) / 2
            mar = detect_mouth(mouth)
            pitch, yaw, roll = get_head_pose(landmarks, (h, w))

            if mar > 0.33:
                predicted_state = 'yawn'
            elif ear < 0.25 and mar < 0.3:
                predicted_state = 'close_eyes'
            
    else:
        stats['no_faces_detected'] += 1
        continue 
    
    stats['true_class_dist'][true_class] += 1
    stats['predicted_dist'][predicted_state] += 1
    stats['true_vs_predicted'][true_class][predicted_state] += 1
    
    
    results.append({
        'filename': img_file,
        'true_class': true_class,
        'predicted_class': predicted_state,
        'EAR': round(ear, 2),
        'MAR': round(mar, 2),
        'yaw': round(yaw, 2),
        'pitch': round(pitch, 2),
        'roll': round(roll, 2),
        'face_detected': face_detected
    })
    

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
unique_classes = sorted(set(df['true_class'].unique()).union(set(df['predicted_class'].unique())))

print(classification_report(df['true_class'], df['predicted_class'], target_names=unique_classes, zero_division=0 ))

print(f"Всего изображений: {stats['total_images']}")
print(f"Найдено лиц: {stats['faces_detected']} ({stats['faces_detected']/stats['total_images']:.1%})")
print(f"Изображений без лиц: {stats['no_faces_detected']}")

print("\nИстинные классы:")
for cls, count in stats['true_class_dist'].items():
    print(f"{cls}: {count} изображений")

print("\nПредсказанные классы:")
for cls, count in stats['predicted_dist'].items(): 
    print(f"{cls}: {count} изображений")
