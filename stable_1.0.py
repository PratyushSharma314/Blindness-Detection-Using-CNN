from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense
from keras import backend as K
import numpy as np
import pandas as pd
from keras.preprocessing import image

 

#######################################################################

img_width, img_height = 299, 299

df = pd.read_csv("D:\\COLLEGE WORK\\SEM 7\\Major I\\train.csv")
print("csv file content: ")
print(df.head())
df["diagnosis"]=df["diagnosis"].apply(lambda x:str(x))
df["id_code"] = df["id_code"].apply(lambda x: x + ".png")
print("After preprocessing of csv file content: ")
print(df.head())



train_data_dir = "D:\\COLLEGE WORK\\SEM 7\\Major I\\dataset\\train_images"
validation_data_dir = "D:\\COLLEGE WORK\\SEM 7\\Major I\\dataset\\train_images"
epochs = 1
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#########################################################################

train_datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255.)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = df[1:3000],
    directory = "D:\\COLLEGE WORK\\SEM 7\\Major I\\dataset\\train_images",
    x_col = "id_code",
    y_col = "diagnosis",
    seed = 42,
    shuffle = True,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = "categorical",
    classes = ["0", "1", "2", "3", "4"])

validation_generator = test_datagen.flow_from_dataframe(
    dataframe = df[3000:],
    directory = "D:\\COLLEGE WORK\\SEM 7\\Major I\\dataset\\train_images",
    x_col = "id_code",
    y_col = "diagnosis",
    target_size = (img_width, img_height),
    batch_size = batch_size,
    shuffle = False,
    seed = 42,
    class_mode = "categorical",
    classes = ["0", "1", "2", "3", "4"])

########################################################################

model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.summary()

model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(5 , activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

##################################################################
          
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(
    generator = train_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = STEP_SIZE_VALID)

model.save_weights('beta_1_try.h5')

##################################################################

img_pred = image.load_img("D:\\COLLEGE WORK\\SEM 7\\Major I\\dataset\\test_images\\0a2b5e1a0be8.png", target_size = (299, 299))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

rslt = model.predict(img_pred)
print("printing predicted test data class: ")
print(rslt)


          








 
    
