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

train = pd.read_csv('/mnt/vol1/Sakshi/Satellite imageclassification/resisc_part2/info.csv', encoding='windows-1252')    # reading the csv file
print(train.head() )     # printing first five rows of the file

print(train.columns)
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('/mnt/vol1/Sakshi/Satellite imageclassification/resisc_part2/images/'+train['Id'][i]+'.jpg',target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

print('x shape',X.shape)
y = np.array(train.drop(['Id', 'Genre'],axis=1))
print('y shape',y.shape)
# plt.plot(X[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='sigmoid'))
model.add(Flatten())
print(model.summary)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), batch_size=64)
model.save('sat_classification_model1.h5')


