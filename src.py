import os

# base data sets directory
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
