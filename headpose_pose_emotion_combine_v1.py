from tensorflow import keras
import time
import mediapipe as mp
import pickle 
import pandas as pd
import cv2
import numpy as np

pred_model = keras.models.load_model('model')
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

#####################################################################################################
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#####################################################################################################

cap = cv2.VideoCapture(0)

predicted_labels = []

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    prev_test = np.zeros((48,48,3))
    
    while cap.isOpened():
        ret, frame = cap.read()
        start = time.time()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
        
        ################################################################################################
        
        # Recolor Feed
        pose_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_image.flags.writeable = False        
        
        # Make Detections
        pose_results = holistic.process(pose_image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        pose_image.flags.writeable = True   
        pose_image = cv2.cvtColor(pose_image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(pose_image, pose_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(pose_image, pose_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(pose_image, pose_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        
        # 4. Pose Detections
        mp_drawing.draw_landmarks(pose_image, pose_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = pose_results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = pose_results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
#             # Append class name 
#             row.insert(0, class_name)
            
#             # Export to CSV
#             with open('coords.csv', mode='a', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow(row) 

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            #print(body_language_class, body_language_prob)
            
            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (pose_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 pose_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            cv2.rectangle(pose_image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(pose_image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(pose_image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(pose_image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(pose_image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(pose_image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(pose_image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', pose_image)
        
    ####################################################################################################    
        headpose_image = frame
        headpose_image.flags.writeable = False
        headpose_results = face_mesh.process(headpose_image)
        headpose_image.flags.writeable = True
        headpose_image = cv2.cvtColor(headpose_image, cv2.COLOR_RGB2BGR)
        
        headpose_img_h, headpose_img_w, headpose_img_c = headpose_image.shape
        face_3d = []
        face_2d = []
        
        if headpose_results.multi_face_landmarks:
            for face_landmarks in headpose_results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * headpose_img_w, lm.y * headpose_img_h)
                            nose_3d = (lm.x * headpose_img_w, lm.y * headpose_img_h, lm.z * 3000)
                            
                        x, y = int(lm.x * headpose_img_w), int(lm.y * headpose_img_h)
    
                         # Get the 2D Coordinates
                        face_2d.append([x, y])
    
                         # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])
                        
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
    
                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)
    
                # The camera matrix
                focal_length = 1 * headpose_img_w
    
                cam_matrix = np.array([ [focal_length, 0, headpose_img_h / 2],
                                        [0, focal_length, headpose_img_w / 2],
                                        [0, 0, 1]])
                
                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
    
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                
                # See where the user's head tilting
                if y < -15:
                    text = "disengaged"
                    #print('disengaged')
                elif y > 15:
                    text = "disengaged"
                    #print('disengaged')
                elif x < -10:
                    text = "disengaged"
                    #print('disengaged')
                elif x > 10:
                    text = "disengaged"
                    #print('disengaged')
                else:
                    text = "engaged"
                    #print('engaged')
    
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                
                cv2.line(headpose_image, p1, p2, (255, 0, 0), 3)
    
                # Add the text on the image
                cv2.putText(headpose_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(headpose_image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(headpose_image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(headpose_image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    
    
            end = time.time()
            totalTime = end - start
    
            #fps = 1 / totalTime
            #print("FPS: ", fps)
    
            #cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
            mp_drawing.draw_landmarks(
                        image=headpose_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            
            cv2.imshow('Head Pose Estimation', headpose_image)
            
    ################################################################################################################       
            
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            prediction = 'disengaged'
            #print('disengaged')
            cv2.putText(frame, 'disengaged', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
                
        else:
            for (x, y, w, h) in faces: # For each detected face:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
                roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
                roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
            #canvas, color_face = detect(gray, frame) # We get the output of our detect function.
        
        
            test_image = cv2.resize(roi_color,(48,48))
            test_image_expand = np.expand_dims(test_image, axis = 0)
            result = pred_model.predict(test_image_expand)
            # comparison = test_image == prev_test
            # if comparison.all():
            #     prediction = 'disengaged'
            # #training_set.class_indices
            if result[0][0] == 1:
                prediction = 'engaged'
            else:
                prediction = 'disengaged'
                
            labels = (prediction, text, body_language_class)
            print(prediction, text, body_language_class)
            prev_test = test_image
            cv2.putText(frame, prediction, (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('frame', frame)
    
            end = time.time()
            totalTime = end - start
    
            fps = 1 / totalTime
            print("FPS: ", fps)
            
            predicted_labels.append(labels)
            
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()

import pandas as pd
df = pd.DataFrame(predicted_labels, columns=['Emotion', 'headpose', 'Pose'])




