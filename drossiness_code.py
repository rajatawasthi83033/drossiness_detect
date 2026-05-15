import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pygame


pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("alarm_sound.mp3")

eye_model=load_model("eye_detect96.keras")
yawn_model=load_model("yawn_model82.keras")


face_detect=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

eye_detect=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_eye.xml"
)

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error:camera not open!")
    exit()

label="no eye"

counter=0

while True:
    ret,frame=cap.read()

    if ret:

        eye_closed=False

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2)

        for x,y,w,h in faces:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)

            roi_gray=gray[y:y+h,x:x+w]

            roi_color=frame[y:y+h,x:x+w]

            eyes=eye_detect.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=5,minSize=(20,20))

            for (ex,ey,ew,eh) in eyes:

                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),thickness=3)

                eye_crop=roi_color[ey:ey+eh,ex:ex+ew]

                eye_crop=cv2.resize(eye_crop,(64,64))

                eye_crop=eye_crop/255.0

                eye_crop = np.reshape(eye_crop,(1,64,64,3))


                pred=eye_model.predict(eye_crop,verbose=0)

                

                if pred>70:

                    label="opened"

                else:
                    label="closed"
                    eye_closed=True

                
                    

        if eye_closed:
            counter=min(counter+1,100)
        else:
            counter=0
            pygame.mixer.music.stop()

        cv2.putText(
        frame,
        f"Counter: {counter}",
        (50,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2)


        if counter>30:
            cv2.putText(
                frame,"Drossiness",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2
                )
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

        else:
            cv2.putText(
                frame,f"Status = {label}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2
                )
            

            

        cv2.imshow("eye_detect",frame)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    else:
        print("not collect frame")
        break

    
    


    
               


                



cap.release()
cv2.destroyAllWindows()






    