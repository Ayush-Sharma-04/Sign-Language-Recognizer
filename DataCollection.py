import os
import cv2

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

no_of_classes = 24
dataset_size = 200

cap = cv2.VideoCapture(0)
for i in range(no_of_classes):
    if not os.path.exists(os.path.join(data_dir, str(i))):
        os.makedirs(os.path.join(data_dir, str(i)))

    print('collecting data for class {}'.format(i))

    done = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Press q to start recording", (50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) ==  ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(i), '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()