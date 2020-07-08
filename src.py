import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------
dataset = 'dataxray'
# data content
train = os.path.join(dataset,'train')
val = os.path.join(dataset,'val')
test = os.path.join(dataset,'test')
# normal and pneumonia separation
train_normal = os.path.join(train,'NORMAL')
train_pneumonia = os.path.join(train, 'PNEUMONIA')
val_normal = os.path.join(val, 'NORMAL')
val_pneumonia = os.path.join(val, 'PNEUMONIA')
test_normal = os.path.join(test, 'NORMAL')
test_pneumonia = os.path.join(test, 'PNEUMONIA')
# number of train images
print('Numer of Training Images')
print('normal case    :', len(os.listdir(train_normal)))
print('pneumonia case :', len(os.listdir(train_pneumonia)))

# -----------------------------------------------------------
# IMGAES AUGMENTATION
# -----------------------------------------------------------
print()
print('image generator')
batch_size = 20
target_size = (224, 224)
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator(rescale = 1.0/255)
validation_datagen = ImageDataGenerator(rescale = 1.0/255)
test_datagen  = ImageDataGenerator(rescale = 1.0/255)
# Specify the image classes
classes = ['NORMAL', 'PNEUMONIA']
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train,
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    classes=classes,
                                                    target_size=target_size)
# Flow validation images in batches of 20 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(val,
                                                              batch_size=batch_size,
                                                              class_mode='binary',
                                                              classes=classes,
                                                              target_size=target_size)
# Flow testing images in batches of 100 using test_datagen generator
test_generator = test_datagen.flow_from_directory(test,
                                                  batch_size=batch_size,
                                                  class_mode='binary',
                                                  classes=classes,
                                                  target_size=target_size)


# -----------------------------------------------------------
# IMAGES PREVIEW
# -----------------------------------------------------------
plt.style.use('seaborn')
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
plt.imshow(fig)


# -----------------------------------------------------------
# MODELS
# -----------------------------------------------------------
import tensorflow as tf
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224,224,3)),
        tf.keras.layers.Dense(1024,activation= 'relu'),
        tf.keras.layers.Dense(512,activation= 'relu'),
        tf.keras.layers.Dense(256,activation= 'relu'),
        tf.keras.layers.Dense(32,activation= 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid'),
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print()
print(model.summary())