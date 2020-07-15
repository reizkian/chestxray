# CHEST X RAY PNEUMONIA PREDICTION
# created 13 July 2020
# --------------------------------------------

# DATA HANDLING
# directory variable
import os
base = 'dataxray'
directory_train = os.path.join(base,'train')
directory_test = os.path.join(base,'test')
directory_val = os.path.join(base,'val')

train_N = os.path.join(directory_train,'NORMAL')
train_P = os.path.join(directory_train,'PNEUMONIA')

test_N = os.path.join(directory_test,'NORMAL')
test_P = os.path.join(directory_test,'PNEUMONIA')

val_N = os.path.join(directory_val,'NORMAL')
val_P = os.path.join(directory_val,'PNEUMONIA')

print('DATASET CHEST XRAY')
print('number of images')
print('-----------------------------------')
print('Train (normal, pneumonia)',len(os.listdir(train_N)), len(os.listdir(train_P)))
print('Test (normal, pneumonia)',len(os.listdir(test_N)), len(os.listdir(test_P)))
print('Val (normal, pneumonia)',len(os.listdir(val_N)), len(os.listdir(val_P)))


# IMAGE AUGMENTATION - DATA GENERATOR
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 20
target_size = (224, 224)
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator(rescale = 1.0/255)
validation_datagen = ImageDataGenerator(rescale = 1.0/255)
test_datagen  = ImageDataGenerator(rescale = 1.0/255)
# Specify the image classes
classes = ['NORMAL', 'PNEUMONIA']
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(directory_train,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=classes,
                                                    target_size=target_size)     
# Flow validation images in batches of 20 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(directory_val,
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            classes=classes,
                                                            target_size=target_size)
# Flow testing images in batches of 100 using test_datagen generator
test_generator = test_datagen.flow_from_directory(directory_test,
                                                  batch_size=batch_size,
                                                  class_mode='binary',
                                                  classes=classes,
                                                  target_size=target_size)

# PREVIEW IMAGES DATA (after augmentation)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import tensorflow as tf
# Obtain one batch of testing images
images, labels = next(train_generator)
labels = labels.astype('int')
# Plot the images in the batch, along with predicted and true labels
nrows = 4
ncols = batch_size / nrows
fig = plt.figure(figsize=(15, 14))
for idx in range(20):
    ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[])
    plt.imshow(images[idx])
    ax.set_title(classes[labels[idx]])


# MODEL & TRAINING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
model = Sequential([
    # Note the input shape is the desired size of the image 224x224 with 3 bytes color
    Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2), 
    Conv2D(64, (3,3), activation='relu'), 
    MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    Flatten(), 
    # 512 neuron hidden layer
    Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('melanoma') and 1 for the other ('not melanoma')
    Dense(1, activation='sigmoid')  
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import Callback, EarlyStopping
# callback
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.70):
            print("\nReached 70% accuracy so cancelling training!")
            self.model.stop_training = True
# callbacks = myCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# train
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    validation_steps=50,
                    callbacks=[early_stopping],
                    verbose=2)