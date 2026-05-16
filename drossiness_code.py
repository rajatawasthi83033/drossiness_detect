import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pygame
import mediapipe as mp

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alarm_sound.mp3")

eye_model=load_model("eye_detect96.keras")
yawn_model=load_model("yawn_model82.keras")


mp_face_mesh=mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def eye_process(eyes_landmarks,window_name):
    eye_points=[]
    padding=20
    
    for idx in eyes_landmarks:
        landmark=face_landmarks.landmark[idx]
                
        x=int(landmark.x*w)
        y=int(landmark.y*h)
        
        eye_points.append((x,y))
        
        cv2.circle(frame,(x,y),2,(0,255,00),-1)
        
    x_min=max(min([p[0] for p in eye_points]) - padding, 0)
    x_max=min(max([p[0] for p in eye_points]) + padding, w)

    y_min=max(min([p[1] for p in eye_points]) - padding, 0)
    y_max=min(max([p[1] for p in eye_points]) + padding, h)
    
    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        (255,0,0),
        2
        )
    
    crop_eye=rgb_frame[y_min:y_max,x_min:x_max]
    
    
    
    if crop_eye.size==0:
        return None
    eye_resize=cv2.resize(crop_eye,(64,64))
    cv2.imshow(window_name, eye_resize)
    
    eye_norm=eye_resize/255.0
    
    eye = np.expand_dims(eye_norm, axis=0)
    
    return eye

def check_eye(predict):
    pred_value = predict[0][0]
    if pred_value>0.55:
        return "opened"
    else:
        return "closed"
    
left_eye = [33,133,160,159,158,144]
right_eye = [362,263,387,386,385,373]


cap=cv2.VideoCapture(0)
counter=0

if not cap.isOpened():
    print("camera is not openning.")
    exit()
    

while True:
    
    ret,frame=cap.read()
    
    if not ret:
        print("Exit")
        break
    
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=face_mesh.process(rgb_frame)
    
    if result.multi_face_landmarks:
        
        for face_landmarks in result.multi_face_landmarks:
            
            left_class = "opened"
            right_class = "opened"
            
            
            h,w,c=frame.shape
            final_left_eye=eye_process(left_eye,"left eye")
            
            final_right_eye=eye_process(right_eye,"right_eye")
            
            if final_left_eye is not None:
                
                left_pred=eye_model.predict(final_left_eye,verbose=0)
                
                left_class=check_eye(left_pred)
                
                cv2.putText(frame,
                            f"L: {left_class}",
                            (20,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2
                            )
            if final_right_eye is not None:
                
                right_pred=eye_model.predict(final_right_eye,verbose=0)
                
                right_class=check_eye(right_pred)
                
                
                cv2.putText(
                    frame,
                    f"R: {right_class}",
                    (20,150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )
                                
            print(left_pred, right_pred)
            
            if right_class=="closed" and left_class=="closed":
                counter+=1
            
            else:
                counter=0
    else:
        counter=0
        print("no face detect")
        
    if counter > 20:

        if not pygame.mixer.music.get_busy():

            pygame.mixer.music.play()

    else:

        pygame.mixer.music.stop()
    
    cv2.putText(
        frame,
        f"Counter: {counter}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )
    cv2.imshow("Final", frame)
    
    


    
    if cv2.waitKey(1) & 0xff==ord("q"):
        print("closed by user")
        break
    
  


cap.release()
cv2.destroyAllWindows()
                
                
            
                
                
            
            
                
            
            
    