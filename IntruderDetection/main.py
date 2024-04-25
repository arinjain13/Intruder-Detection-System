import face_recognition
import numpy as np
import cv2 
from playsound import playsound
import pickle
import cvzone


video_capture =cv2.VideoCapture('/Users/apple/Documents/Pythonpro/alert /IntruderDetection/source/vid.mov')

with open('/Users/apple/Documents/Pythonpro/alert /IntruderDetection/source/IDS', 'rb') as f:
    posList = pickle.load(f)

width, height = 160,200

def checkParkingSpace(imgPro):
    Counter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y+height, x:x+width]
        #cv2.imshow(str(x*y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count >21600:
            color = (0,255,0)
            thickness = 5
            Counter +=1

        else:
            color = (0,0,255)
            thickness = 2
            
        cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(frame,str(count),(x,y+height-3), scale = 1, thickness = 2, offset =0,colorR = color)
    if Counter ==1:
        playsound('/Users/apple/Documents/Pythonpro/alert /IntruderDetection/utils/alert.wav')

avish_image=face_recognition.load_image_file("/Users/apple/Documents/Pythonpro/alert /IntruderDetection/known_peoples/avish.jpg")
arin_image = face_recognition.load_image_file("/Users/apple/Documents/Pythonpro/alert /IntruderDetection/known_peoples/arin.jpg")


while True:
    avish_face_encoding=face_recognition.face_encodings(avish_image)[0]
    arin_face_encoding = face_recognition.face_encodings(arin_image)[0]
    break


known_face_encodings = [
    arin_face_encoding,
    avish_face_encoding

]
known_face_names = [
    "arin",
    "avish"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    imgGray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    #rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if (name == "Unknown"):
            print (name, " was here")
            if video_capture.get(cv2.CAP_PROP_POS_FRAMES) == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5) 
            imgMedian = cv2.medianBlur(imgThreshold,5)
            kernel = np.ones((3,3),np.uint8)
            imgDilate = cv2.dilate(imgMedian,kernel,iterations=1)

            checkParkingSpace(imgDilate)
            cv2.waitKey(50)
        

    # Display the resulting image
    cv2.imshow('Video', frame)

        
    #for pos in posList:
    #cv2.imshow("ImageBlur" ,imgBlur)
    #cv2.imshow("ImageThres" ,imgMedian)




    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

