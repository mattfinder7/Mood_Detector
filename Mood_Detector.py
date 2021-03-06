#https://github.com/opencv/opencv/tree/master/data/haarcascades
#https://docs.opencv.org/4.x/
#https://github.com/lilipads/emotion-detection
#https://viso.ai/computer-vision/deepface/
#opacity --> https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle


from cgitb import text
import cv2
from cv2 import FILLED
from deepface import DeepFace
import time
import pyttsx3

#wait at color for 2500 milliseconds
def wait(timer, text):    
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(2000)
    text_to_speech(text)

#output text as audio   
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

text = "Hello world"

#load some pretrained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture video from webcam
webcam = cv2.VideoCapture(0)

#https://stackoverflow.com/questions/39953263/get-video-dimension-in-python-opencv
if webcam.isOpened(): 
    # get webcam property 
    webcam_width  = webcam.get(3)  # float `width`
    webcam_height = webcam.get(4)  # float `height`

    #https://www.geeksforgeeks.org/how-to-convert-float-to-int-in-python/ 
    # change from float to int  
    ww = int(webcam_width)
    wh = int(webcam_height)

timer = time.time()
addtime = 10
#iterate over the frames forever
while True:
    #read current frame
    #successfull_frame_read will always be true
    successfull_frame_read, frame = webcam.read()
    result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
    
    #convert to greyscale 
    #doc: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    #doc: https://docs.opencv.org/4.x/d1/de5/classcv_1_1CascadeClassifier.html
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
 
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)#color is BGR not RGB!!!
        print(result['dominant_emotion'])
        if timer <= time.time():
            if result['dominant_emotion'] == 'happy':
                #make rectangle transparent
                #frame[0:wh, 0:ww, :] = cv2.addWeighted(frame[0:wh, 0:ww, :], 0.5, frame[0:wh, 0:ww, :], 0.5, 0)
                cv2.rectangle(frame, (0,0), (ww, wh), (0, 255, 0), FILLED)
                
                #wait at color for 2500 milliseconds and don't select another color for 'addtime' seconds
                text = "Happy"
                wait(text) 
                timer = time.time() + addtime
                
            elif result['dominant_emotion'] == 'angry':
                cv2.rectangle(frame, (0,0), (ww, wh), (0, 0, 255), FILLED)
                wait()
                timer = time.time() + addtime
            elif result['dominant_emotion'] == 'sad':
                cv2.rectangle(frame, (0,0), (ww, wh), (255, 0, 0), FILLED)
                wait()
                timer = time.time() + addtime
            elif result['dominant_emotion'] == 'disgust':
                cv2.rectangle(frame, (0,0), (ww, wh), (255, 140, 0), FILLED)
                wait()
                timer = time.time() + addtime
            elif result['dominant_emotion'] == 'fear':
                cv2.rectangle(frame, (0,0), (ww, wh), (52, 25, 48), FILLED)
                wait()
                timer = time.time() + addtime
            elif result['dominant_emotion'] == 'surprise':
                cv2.rectangle(frame, (0,0), (ww, wh), (0, 225, 225), FILLED)
                wait()
                timer = time.time() + addtime

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #the key to quit
    #81=Q   #113=q
    if key==81 or key==113:
        break

#ends the webcam
webcam.release()

print("Done")

