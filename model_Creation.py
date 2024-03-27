
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (12,12)

face_images_db = np.load('Data\\Faces\\face_images.npz')['face_images']
facial_keypoints_df = pd.read_csv('Data\\Faces\\facial_keypoints.csv')

(im_height, im_width, num_images) = face_images_db.shape
num_keypoints = facial_keypoints_df.shape[1] / 2

print('number of images = %d' %(num_images))
print('image dimentions = (%d,%d)' %(im_height,im_width))
print('number of facial keypoints = %d' %(num_keypoints))




rects = []
images = []
for i in range(num_images):
    #[x,width,y,length]
    if(  (not np.any(np.isnan(facial_keypoints_df.iloc[i,:]["right_eye_center_y"])) and not np.any(np.isnan(facial_keypoints_df.iloc[i,:]["right_eye_center_x"])) and not np.any(np.isnan(facial_keypoints_df.iloc[i,:]["left_eye_center_y"])) and not np.any(np.isnan(facial_keypoints_df.iloc[i,:]["left_eye_center_x"])) and not np.any(np.isnan(facial_keypoints_df.iloc[i,:]["mouth_center_bottom_lip_y"])))):
        rect = []
        rect.append(facial_keypoints_df.iloc[i,:]["right_eye_center_x"])
        rect.append(facial_keypoints_df.iloc[i,:]["left_eye_center_x"] - facial_keypoints_df.iloc[i,:]["right_eye_center_x"])
        rect.append(facial_keypoints_df.iloc[i,:]["right_eye_center_y"])
        rect.append(facial_keypoints_df.iloc[i,:]["mouth_center_bottom_lip_y"] - facial_keypoints_df.iloc[i,:]["right_eye_center_y"])
        image = []
        for j in face_images_db[:,:,i]:
            image.extend(j)

        images.append(image)

        rects.append(rect)


    



from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import sparse_categorical_crossentropy
from keras.losses import mean_squared_logarithmic_error
import tensorflow as tf
#load the dataset

x = tf.convert_to_tensor(images, dtype=tf.float32)

y = tf.convert_to_tensor(rects, dtype=tf.float32)


#define the model
model = Sequential()
model.add(Dense(64,activation = 'relu'))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(4))

#compile the model
model.compile(loss= mean_squared_logarithmic_error, optimizer='adam', metrics=['accuracy'])

#fit the model to the dataset (the real network thingy)
model.fit(x,y, epochs = 200, batch_size = 50)
model.save("picture_model2.keras")

#determine accuracy
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))



