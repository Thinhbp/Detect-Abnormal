from tensorflow.keras.models import load_model
import numpy as np
import cv2

def mean_squared_loss(x1,x2):
    difference=x1-x2
    sq_difference=difference**2
    s=sq_difference.sum()
    distance=np.sqrt(s)
    mean_distance=np.mean(distance)
    return mean_distance

model=load_model('saved_model.h5')

cap = cv2.VideoCapture(0)
index = 1

while(True):
    ret, frame = cap.read()
    pixel=[]
    for i in range(10):
        ret, frame = cap.read()
        frame  = cv2.resize(frame , dsize=None, fx=0.9, fy=0.9)
        image = frame.copy()
        image = cv2.resize(image, dsize=(227, 227))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixel.append(image)
    pixel=np.array(pixel)
    pixel = pixel.astype('float32') / 255
    pixel= pixel[:10, :, :]
    pixel = pixel.reshape(-1,227, 227, 10)
    pixel = np.expand_dims(pixel, axis=4)
    prediction=model.predict(pixel)
    loss=mean_squared_loss(prediction,pixel)
    print(loss)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if loss >= 355:
        cv2.putText(frame, "Abnormal Event", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
        print("Abnormal Event")
    else:
        cv2.putText(frame, "normal Event", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255,0), 1)
        print("normal Event")
    cv2.imshow("video", frame)



cap.release()
cv2.destroyAllWindows()






