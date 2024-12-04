import os
import tensorflow as tf
from preprocessing import DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Activation,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import json

loader=DataLoader()
loader.load_data()
cls=loader.get_classes()

#base model from the pretrained model 
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable=False

#Model
model=Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Flatten(),
        Dropout(0.3),
        Dense(123),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(len(cls),activation='softmax')
    ]
)

model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
history=model.fit(loader.train_data,epochs=20,validation_data=loader.test_data,batch_size=30)

#save model
model.save('./trained models/fruit_classifier_model.h5')
print('model saved succesfully')

#save history
with open('trained models/classifier_history.json','w') as file:
    json.dump(history.history,file)