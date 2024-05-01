#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq
import numpy as np

import cv2
from cvzone.HandTrackingModule import HandDetector

cv2.setNumThreads(4)

# Parameters
width, height = 1280, 720

# Hand Detector
detector = HandDetector(maxHands=2, detectionCon=0.8)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Handedness
handType, handType1 = "", ""

while True:
    #  Wait for next request from client
    message = socket.recv()

    # Convert ByteArray data from Unity into numpy array for OpenCV
    image_byte_array = bytearray(message)
    np_array = np.asarray(image_byte_array, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)

    # Hands
    hands, img = detector.findHands(img)

    data = []
    data1 = []
    
    # Landmark values - {x, y, z} * 21
    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list
        lmList = hand['lmList']
        # Get the handedness
        handType = hand["type"]

        # print(lmList)
        for lm in lmList:
            data.extend([lm[0],height - lm[1],lm[2]])
        # print(data)
            #socket.send(str.encode(str(data)))
        
        # Check if a second hand is detected
        if len(hands) == 2:
            # Information for the second hand
            hand1 = hands[1]
            # Get the landmark list
            lmList1 = hand1["lmList"]
            # Get the handedness
            handType1 = hand1["type"]

            for lm in lmList1:
                data1.extend([lm[0],height - lm[1],lm[2]])

    #img = cv2.resize(img, (0,0), None, 0.5,0.5)
    #cv2.imshow("Image", img)
    # cv2.waitKey(1)

    #  Send reply back to client
    #  In the real world usage, after you finish your work, send your output here
    # socket.send(b"World")
    socket.send(str.encode(handType + str(data) + "|" + handType1 + str(data1)))
