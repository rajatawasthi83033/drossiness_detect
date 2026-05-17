def eye_process(eyes_landmarks):
    eye_points=[]
    padding=20
    
    for idx in eyes_landmarks:
        landmark=face_landmarks.landmark[idx]
                
        x=int(landmark.x*w)
        y=int(landmark.y*h)
        
        eye_points.append((x,y))
        
        # cv2.circle(frame,(x,y),2,(0,255,00),-1)
        
    x_min=max(min([p[0] for p in eye_points]) - padding, 0)
    x_max=min(max([p[0] for p in eye_points]) + padding, w)

    y_min=max(min([p[1] for p in eye_points]) - padding, 0)
    y_max=min(max([p[1] for p in eye_points]) + padding, h)

    
    crop_eye=rgb_frame[y_min:y_max,x_min:x_max]
    crop_eye = cv2.convertScaleAbs(crop_eye, alpha=1.2, beta=20)

    eye_resize = cv2.resize(crop_eye, (64,64))

    eye_norm = eye_resize / 255.0

    eye = np.expand_dims(eye_norm, axis=0)


    return eye, eye_resize, (x_min, y_min, x_max, y_max)

def check_eye(predict):
    pred_value = predict[0][0]
    if pred_value>0.40:
        return "opened"
    else:
        return "closed"
    

    
def draw_eye_box(eye_points,eye_class):
    x_min,y_min,x_max,y_max=eye_points
    color = (0,255,0) if eye_class=="opened" else (0,0,255)
    cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),color,2)



def distance(p1,p2):
    dis=math.sqrt(((p2.x-p1.x)**2)+((p2.y-p1.y)**2))
    return dis


def calculate_EAR(landmarks,eye_points):
    p1=landmarks[eye_points[0]]
    p2=landmarks[eye_points[1]]
    p3=landmarks[eye_points[2]]
    p4=landmarks[eye_points[3]]
    p5=landmarks[eye_points[4]]
    p6=landmarks[eye_points[5]]


    vartical1=distance(p2,p6)
    vartical2=distance(p3,p5)
    horizontal=distance(p1,p4)

    EAR=(vartical1+vartical2)/(2.0*horizontal)

    return EAR




import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pygame
import mediapipe as mp
import math

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alarm_sound.mp3")

eye_model=load_model("eye_detect97.keras")
# yawn_model=load_model("yawn_model82.keras")


mp_face_mesh=mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)


    
left_eye = [33,160,158,133,153,144]
right_eye = [362,385,387,263,373,380]


cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

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
            
            left_EAR=calculate_EAR(face_landmarks.landmark,left_eye)
            right_EAR=calculate_EAR(face_landmarks.landmark,right_eye)

            # print(f"LEFT EAR = {left_EAR}, RIGHT EAR = {right_EAR}")




            h,w,c=frame.shape
            final_left_eye,left_eye_size,left_eye_points=eye_process(left_eye)
            
            final_right_eye,right_eye_size,right_eye_points=eye_process(right_eye)
            

            

            
            
            if final_left_eye is not None and final_right_eye is not None:
                
                left_eye_size = cv2.resize(left_eye_size, (150,150))
                right_eye_size = cv2.resize(right_eye_size, (150,150))
                
                combined = np.hstack((left_eye_size, right_eye_size))
                cv2.imshow("Both Eyes", combined)
            
            
            if final_left_eye is not None:
                
                left_pred=eye_model.predict(final_left_eye,verbose=0)
                
                left_class=check_eye(left_pred)
                
                # draw_eye_box(left_eye_points,left_class)

                
                left_confidence = left_pred[0][0]
                # cv2.putText(
                #     frame,
                #     f"L: {left_class} {left_confidence:.2f}",
                #     (20,100),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0,255,0),
                #     2)
            
            if final_right_eye is not None:
                
                right_pred=eye_model.predict(final_right_eye,verbose=0)
                
                right_class=check_eye(right_pred)
                
                # draw_eye_box(right_eye_points,right_class)
                
                
                right_confidence = right_pred[0][0]
                # cv2.putText(
                #     frame,
                #     f"R: {right_class} {right_confidence:.2f}",
                #     (20,150),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0,255,0),
                #     2)
                                
            # if final_left_eye is not None and final_right_eye is not None:
            #     print(left_pred, right_pred)



            model_pred=(right_class=="closed" and left_class=="closed")

            avg_EAR = (left_EAR + right_EAR) / 2

            if model_pred or avg_EAR < 0.22:
                counter += 1
            else:
                counter = 0
    
    else:
        counter=0
        print("no face detect")
        
        
    if counter > 5:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    else:
        pygame.mixer.music.stop()
        
    
    # cv2.putText(
    #     frame,
    #     f"Counter: {counter}",
    #     (20,50),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0,0,255),
    #     2
    # )
    
    cv2.imshow("Final", frame)
    

    if cv2.waitKey(1) & 0xff==ord("q"):
        print("closed by user")
        break
    
  
cap.release()
cv2.destroyAllWindows()
                
                
            
                
                
            
            
                
            
            
    