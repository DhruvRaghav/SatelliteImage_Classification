import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# %matplotlib inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path='/home/prateek/Downloads/Sat_classification_test_data/others'
for img1 in os.listdir(path):
    # print(os.path.join(path,img1))
    img = image.load_img(os.path.join(path,img1),target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    model=keras.models.load_model('sat_classification_model.h5')
    classes = ['Dense_Residential', 'Sparse_Residential',
           'Medium_Residential', 'River', 'Forest', 'Highway', 'Industrial']
    proba = model.predict(img.reshape(1,400,400,3))
    top_3 = np.argsort(proba[0])[:-6:-1]
    result=''
    for i in range(5):
        if proba[0][top_3[i]]>0.001:
            if (result == ''):
                result = str(classes[top_3[i]])
            else:
                result = result + ' , ' + str(classes[top_3[i]])
            # print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
            # print("{}".format(classes[top_3[i]]))
    print(img1 + '---'+ result)
# plt.imshow(img)
# # view raw
