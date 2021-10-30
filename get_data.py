import os
import numpy as np
import cv2

path='Abnormal'
pixel=[]
for video in os.listdir(path):
    vid = cv2.VideoCapture(os.path.join(path,video))
    index = 0
    while(True):
        # Extract images
        ret, frame = vid.read()
        # end of frames
        if not ret:
            break
        if index%5==0:
            image=cv2.resize(frame,dsize=(227,227))
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pixel.append(image)
        index += 1

pixel=np.array(pixel)
pixel=pixel.astype('float32')/255

np.save('training.npy',pixel)