import numpy as np
import pickle
import cv2
import mediapipe as mp

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles =mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

label_dict = {0: 'A', 1:'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y'}

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        for hand_landmarks in result.multi_hand_landmarks:

            for i in range(len(hand_landmarks.landmark)):
                   x = hand_landmarks.landmark[i].x
                   y = hand_landmarks.landmark[i].y
                   data_aux.append(x)
                   data_aux.append(y)
                   x_.append(x)
                   y_.append(y)
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * W) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * W) - 10

        prediction = model.predict([(np.asarray(data_aux))])
        predicted_char = label_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1 -10), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, predicted_char, (x1, y1 -20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,(0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)
    cv2.waitKey(25)

    if cv2.waitKey(25) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
