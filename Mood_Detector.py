#https://github.com/opencv/opencv/tree/master/data/haarcascades
#https://docs.opencv.org/4.x/
#https://github.com/lilipads/emotion-detection
#https://viso.ai/computer-vision/deepface/


import cv2
from cv2 import FILLED
from deepface import DeepFace

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

    #opacity --> https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
    for (x,y,w,h) in face_coordinates:
        print(result['dominant_emotion'])
        if result['dominant_emotion'] == 'happy':
            cv2.rectangle(frame, (0,0), (ww, wh), (0, 255, 0), FILLED)
        elif result['dominant_emotion'] == 'angry':
            cv2.rectangle(frame, (0,0), (ww, wh), (0, 0, 255), FILLED)
        elif result['dominant_emotion'] == 'sad':
            cv2.rectangle(frame, (0,0), (ww, wh), (255, 0, 0), FILLED)
        elif result['dominant_emotion'] == 'disgust':
            print('disgust')
        elif result['dominant_emotion'] == 'fear':
            print('fear')
        elif result['dominant_emotion'] == 'surprise':
            cv2.rectangle(frame, (0,0), (ww, wh), (255, 225, 0), FILLED)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #the key to quit
    #81=Q   #113=q
    if key==81 or key==113:
        break

#ends the webcam
webcam.release()



#will show the img
#cv2.imshow('Face Dector', frame)
#cv2.waitKey() #without img will only show for 1 sec and then close

print("Done")