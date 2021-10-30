from tensorflow.keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

train_data=np.load('training.npy')
a=train_data.shape[0]-train_data.shape[0]%10
train_data=train_data[:1000,:,:]
print(train_data.shape)

model=Sequential()
model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='relu'))
model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='relu'))
model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',return_sequences=True))
model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same'))
model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='relu'))
model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='relu'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

train_data=train_data.reshape(-1,227,227,10)
train_data=np.expand_dims(train_data,axis=4)

epochs=5
batch_size=1
file="weights-{epoch:02d}-{accuracy:.2f}.hdf5"

callback = ModelCheckpoint(file, monitor="accuracy", verbose=1, save_best_only=True, mode='max')


model.fit(train_data,train_data, batch_size=batch_size, epochs=epochs, callbacks = [callback])
model.save("saved_model.h5")


