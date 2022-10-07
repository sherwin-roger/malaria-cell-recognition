# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.
Dataset:CellImage

## Neural Network Model
![image](https://user-images.githubusercontent.com/75235128/194599874-887d11d1-f97a-45fd-9328-3ff6202f6926.png)

## DESIGN STEPS

### STEP 1:
Import necessary packages
### STEP 2:
Preprocess the image using data augmentation
### STEP 3:
Fit the model using the augmented images

## PROGRAM
```
python3
!git clone https://github.com/sherwin-roger/cellimage.git

import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

trainDatagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
testDatagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train=trainDatagen.flow_from_directory("cellimage/cell_images/train",class_mode = 'binary',target_size=(150,150))
test=trainDatagen.flow_from_directory("cellimage/cell_images/test",class_mode = 'binary',target_size=(150,150))

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,activation="relu",padding="same"),
    tf.keras.layers.Conv2D(32,3,activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.Conv2D(64,3,activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(128,3,activation="relu"),
    tf.keras.layers.Conv2D(128,3,activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="accuracy")

model.fit(train,epochs=5,validation_data=test)

pd.DataFrame(model.history.history).plot()

import numpy as np

test_predictions = np.argmax(model.predict(test), axis=1)

confusion_matrix(test.classes,test_predictions)

print(classification_report(test.classes,test_predictions))

import numpy as np

img = tf.keras.preprocessing.image.load_img("cellimage/cell_images/test/uninfected/C100P61ThinF_IMG_20150918_144104_cell_34.png")
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(150,150))
img_28 = img_28/255.0
img_28=tf.expand_dims(img_28, axis=0)

if tf.cast(tf.round(model.predict(img_28))[0][0],tf.int32).numpy()==1:
  print("uninfected")
else:
  print("parasitized")
```
Include your code here

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235128/194602885-bd64fbcb-b691-4020-bd01-537927b9c6b8.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235128/194602953-7184d66e-2584-432c-8a7d-60017ef65cc2.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235128/194603006-5e8bfc3a-7f43-4a2b-9a31-070a8e6b0afb.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235128/194603099-2ce14338-1a4c-4e2d-ae04-a9da29a89e12.png)

## RESULT
A deep neural network for Malaria infected cell recognition is built 
