import os
import cv2
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles =mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data_dir = './data'

data = []
labels =[]

for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        data_aux =[]
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                # mp_drawing.draw_landmarks(      #uncomment to see hand landmarks (code block 1)
                #     imgRGB,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                # )
    #          plt.figure()                  #uncomment to see hand landmarks (code block 2)
    #          plt.imshow(imgRGB)
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)


#  plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

