import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import serial
import time

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Initialize the serial port with the correct COM port and baud rate
ser = serial.Serial('COM11', 9600)  # Adjust 'COM3' and baud rate as per your Arduino setup

while True:
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]
        print("Prediction:", pred)

        if(pred=='Happy'):
            # Send the prediction to the actual serial port
            ser.write(b'h')
            print("Prediction sent to serial port")    
        
        elif(pred=="Sad"):
            ser.write(b's')
            print("Prediction sent to serial port")    
            
        elif(pred=="Neutral"):
            ser.write(b'n')
            print("Prediction sent to serial port")    
            
        elif(pred=="Anger"):
            ser.write(b'a')
            print("Prediction sent to serial port") 
               
        elif(pred=="Disguist"):
            ser.write(b'w')
            print("Prediction sent to serial port")    
            
        elif(pred=="Drowsy"):
            ser.write(b'd')
            print("Prediction sent to serial port")    
        
        elif(pred=="Fear"):
            ser.write(b'f')
            print("Prediction sent to serial port")    
            
        
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    # Delay for 5 seconds
    time.sleep(4)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        ser.close()
        break
