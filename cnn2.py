from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.preprocessing import image
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Initialing the CNN
classifier = Sequential()

#Step 1 - Convolution
#Extract features from the images
classifier.add(Conv2D(32, (3, 3),
                     input_shape = (64, 64, 3),
                     activation = 'relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

#Fitting the CNN to the images

#Generate the training set
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

#Only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

#Process the data
training_set = train_datagen.flow_from_directory('covid_train',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
test_set = test_datagen.flow_from_directory('covid_test',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')      

#Training Model
#Generate the data into CNN
#Execute the Neural Network
classifier.fit_generator(training_set,
                        steps_per_epoch=100,
                        epochs=5,
                        validation_data=test_set,
                        validation_steps=10)            

#Making Predictions
test_image = image.load_img('covid_test/Positif/covid-19-rapidly-progressive-acute-respiratory-distress-syndrome-ards-day-2.jpg', target_size= (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)  
test_set.class_indices
if result[0][0] == 1:
    prediction = 'Positif'
else:
    prediction = 'Negatif'
print(prediction)