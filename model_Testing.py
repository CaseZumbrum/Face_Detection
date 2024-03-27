from keras.models import load_model, Sequential

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (12,12)

face_images_db = np.load('Data\\Faces\\face_images.npz')['face_images']
facial_keypoints_df = pd.read_csv('Data\\Faces\\facial_keypoints.csv')

(im_height, im_width, num_images) = face_images_db.shape
num_keypoints = facial_keypoints_df.shape[1] / 2


rects = []
images = []
for i in range(1):
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




model = load_model("picture_model.keras")

x = tf.convert_to_tensor(images, dtype=tf.float32)

y = tf.convert_to_tensor(rects, dtype=tf.float32)



print(model.predict(x))
print(y[0])