import socket 
import threading 

from base64 import b64decode

import micropip
micropip.install("numpy")

import numpy as np

import cv2
from cvzone.HandTrackingModule import HandDetector

cv2.setNumThreads(4)

# Parameters
width, height = 1280, 720

# Hand Detector
detector = HandDetector(maxHands=2, detectionCon=0.8)

bind_ip = "127.0.0.1" 
bind_port = 5555

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
# we tell the server to start listening with
# a maximum backlog of connections set to 5
server.listen(5)

print(f"[+] Listening on port {bind_ip} : {bind_port}")            

#client handling thread
def handle_client(client_socket):
    while True:
        #printing what the client sends
        request = client_socket.recv(50000)
        #print(f"[+] Recieved: {request}")

        #sending back the packet
        if detectHand(request, "", "") is not None:
            client_socket.send(detectHand(request, "", "").encode())

def detectHand(message, handType, handType1):

    JPEG = b64decode(message)
    img = cv2.imdecode(np.frombuffer(JPEG,dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
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

            for lm in lmList:
                data.extend([lm[0],height - lm[1],lm[2]])

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
        
        return(handType + str(data) + "|" + handType1 + str(data1))

while True:
    # When a client connects we receive the
    # client socket into the client variable, and
    # the remote connection details into the addr variable
    client, addr = server.accept()
    print(f"[+] Accepted connection from: {addr[0]}:{addr[1]}")

    #spin up our client thread to handle the incoming data
    client_handler = threading.Thread(target=handle_client, args=(client,))
    client_handler.start()