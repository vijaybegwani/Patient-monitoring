import cv2
import numpy as np
import requests

cap = cv2.VideoCapture(0)

#address = "http://192.168.31.111:8080/video"
#cap.open(address)

eyeCascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')  #eye detect model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #face detect model

while True:

    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # if _:

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        hsv_frame,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(1, 1),
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    # print("Found {0} faces!".format(len(faces)))
    # if len(faces) > 0:
    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # frame_tmp = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
    # frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
    eyes = eyeCascade.detectMultiScale(
        hsv_frame,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(1, 1),
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )










    # Red color
    low_red = np.array([170, 70, 0])
    high_red = np.array([190, 255,255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    red_coord = cv2.findNonZero(red_mask)

    #orange colour
    #low_orange = np.array([5, 210, 120])
    #high_orange = np.array([15, 255, 255])
    #orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    #orange = cv2.bitwise_and(frame, frame, mask=orange_mask)
    #orange_coord = cv2.findNonZero(orange_mask)

    # Blue color
    low_blue = np.array([95, 55, 120])
    high_blue = np.array([100, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    blue_coord = cv2.findNonZero(blue_mask)




    # Green color
    low_green = np.array([50, 40, 80])
    high_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    green_coord = cv2.findNonZero(green_mask)



    # Yellow colour
    low_yellow = np.array([25, 30, 120])
    high_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    yellow_coord = cv2.findNonZero(yellow_mask)



     #cyan colour
    low_cyan = np.array([85, 50 , 50])
    high_cyan= np.array([98, 150, 255])
    cyan_mask = cv2.inRange(hsv_frame, low_cyan, high_cyan)
    cyan = cv2.bitwise_and(frame, frame, mask=cyan_mask)
    cyan_coord = cv2.findNonZero(cyan_mask)

    # dark blue Color
    low_darkBlue = np.array([110, 60, 0])
    high_darkBlue = np.array([130, 255, 255])
    darkBlue_mask = cv2.inRange(hsv_frame, low_darkBlue, high_darkBlue)
    darkBlue = cv2.bitwise_and(frame, frame, mask=darkBlue_mask)
    darkBlue_coord = cv2.findNonZero(darkBlue_mask)


    # PInk Colour
    low_pink = np.array([140, 60, 0])
    high_pink = np.array([160, 255, 255])
    pink_mask = cv2.inRange(hsv_frame, low_pink, high_pink)
    pink = cv2.bitwise_and(frame, frame, mask=pink_mask)
    pink_coord = cv2.findNonZero(pink_mask)




    # Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)




    # To find mean of coordinates
    a = np.mean(red_coord, axis=0)
    b = np.mean(green_coord, axis=0)
    c = np.mean(blue_coord, axis=0)
    d = np.mean(yellow_coord, axis=0)
    e = np.mean(pink_coord, axis=0)
    f = np.mean(darkBlue_coord, axis=0)
    g = np.mean(cyan_coord, axis=0)



    #Euclidean Distance
    dist = np.linalg.norm(a - b)
    dist1 = np.linalg.norm(c - b)
    dist2 = np.linalg.norm(a - d)
    dist3 = np.linalg.norm(c - d)
    dist4 = np.linalg.norm(a - e)
    dist5 = np.linalg.norm(c - e)
    dist6 = np.linalg.norm(a - f)
    dist7 = np.linalg.norm(c - f)
    dist8= np.linalg.norm(a - g)
    dist9 = np.linalg.norm(c - g)




    #print (dist)
    #print (dist1)
    #print (dist2)
    #print (dist3)
    #print(dist4)
    #print(dist5)
    # print(dist6)
    # print(dist7)
    # print(dist8)
    # print(dist9)




    cv2.imshow("Frame", frame)
    cv2.imshow("Red", red)
    cv2.imshow("Blue", blue)
    cv2.imshow("Green", green)
    cv2.imshow("Yellow",yellow)
    cv2.imshow("Darkblue", darkBlue)
    cv2.imshow("pink", pink)
    cv2.imshow("Cyan", cyan)
    ##cv2.imshow("Orange", orange)

    #cv2.imshow("Result",result)

    key = cv2.waitKey(1)





    if len(eyes) == 0:


         print("")

    else  :
           #print("Eyes")
           #print("patient is not sleeping")
           if ((dist < 60)) and ((dist1 < 70))  :
               print("The Distance between hand and chest is", dist)
               print("Alert")
               print("Patient may have Chest Pain")
               continue


           elif ((dist2 < 80)) and ((dist3 <80))  :
               print("The Distance between hand and Stomach is", dist2)
               print("Alert")
               print("Patient may have Stomach Ache")
               continue

           elif ((dist4 < 80)) and ((dist5 < 100))  :
               # print("The Distance between hand and Head is" , dist5)
               print("Alert")
               print("Patient may have  Headache ")
               continue

           elif ((dist6 < 50) or (dist7 < 50)) :
               print("The Distance between hand and Shoulder is", dist7)
               print("Alert")
               print("Patient may have Shoulder Problem")
               continue
           """
           elif ((dist8 < 20) or (dist9 < 20)) :
               print("The Distance between hand and Knee is", dist8)
               print("Alert")
               print("Patient may have  Knee Pain ")
               continue
           """


           #elif key == 27 :
               #break
#elif len(eyes) == 0:
    #print("No eyes")
            #frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
            #cv2.imshow('Face Recognition', frame_tmp)





