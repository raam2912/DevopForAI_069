import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  
import tensorflow
from keras import datasets, layers, models


(train_images, train_label),(test_images, test_labels)= datasets.cifar10.load_data()
train_images, test_images = train_images/255 , test_images/255

class_name =['Plane','Car','Bird',' Cat','Deer','Dog','Frog','Horse','Ship','Truck']


model=models.load_model('image_classifier.model.keras')    

img = cv.imread('Deer.jpg')
img =cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_resized= cv.resize(img,(32,32))

plt.imshow(img_resized,cmap=plt.cm.binary)

prediction=model.predict(np.array([img_resized])/255)
index =np.argmax(prediction)

print(f"Prediction is {class_name[index]}")