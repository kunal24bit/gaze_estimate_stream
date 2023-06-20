import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_IRIS = [474, 475, 476, 477]

LEFT_IRIS = [469, 470, 471, 471]

L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]
L_H_UP = [159]
L_H_DOWN = [145]
R_H_UP = []
R_H_DOWN = []


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

def eye_position(iris_center, right_point, left_point, threshold):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    iris_position = ""
    if ratio <=threshold:
        iris_position = "Looking away from the screen(right)"
    elif ratio >threshold and ratio <= threshold + 0.15:
        iris_position = "Looking at the screen"
    else:
        iris_position = "Looking away from the screen(left)"
    return iris_position, ratio

def process_frame(frame, threshold):
    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
        
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_h, img_w = frame.shape[:2]

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype = np.int32)
            center_right = np.array([r_cx, r_cy], dtype = np.int32)

            cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
            #cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255),-1, cv2.LINE_AA)
            #cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255),-1, cv2.LINE_AA)
            #cv2.circle(frame, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255),-1, cv2.LINE_AA)
            #cv2.circle(frame, mesh_points[L_H_LEFT][0], 3, (0, 255, 255),-1, cv2.LINE_AA)
            iris_position, ratio = eye_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0], threshold)

            return iris_position, ratio
        
        return "No Face Detected", "0.0"




