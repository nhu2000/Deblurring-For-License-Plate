#Date:2017.4.15
#Version: Python2.7
#The configuration of neural network used here refers to paper CNN FOR LICENSE PLATE MOTION DEBLURRING, Pavel Svoboda, Michal Hradis ...

from __future__ import print_function
import keras
import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D
from keras.models import save_model, load_model
from keras import backend

# input image dimensions
img_rows, img_cols = 66, 66

#x_train, a0001.png~a2015.png, (2015, 128, 264, 3)
#y_train, a0001.png~a2015.png, (2015, 128, 264, 3)
#x_test, b0001.png~b1590.png, (1590, 128, 264, 3)
#y_test, b0001.png~b1590.png, (1590, 128, 264, 3)

#generate string of file path
#i.e. file_path('blur/I1_', i, '.png')
def num3(i):
	if (i > 99):
		addc = ''
	elif (i > 9):
		addc = '0'
	else:
		addc = '00'
	return addc + str(i)	

def num4(i):
	if (i > 999):
		addc = ''
	elif (i > 99):
		addc = '0'
	elif (i > 9):
		addc = '00'
	else:
		addc = '000'
	return addc + str(i)	

#convert form (x,y,rgb) to (rgb, x, y)
def cvt(x): #the form of x is (x, y, rgb)
	r = x[:,:,0]
	g = x[:,:,1]
	b = x[:,:,2]
	r = r.reshape((1,) + r.shape)
	g = g.reshape((1,) + g.shape)
	b = b.reshape((1,) + b.shape)
	return np.concatenate((r, g, b))

#input_data, convert img to array

#x_train
for i in range(1, 2016):
	img = load_img('blur/a' + num4(i) + '.png')
	if (i == 1):
		x_train = img_to_array(img)
		#x_train = cvt(x_train)
		x_train = x_train.reshape((1,) + x_train.shape)
	else:
		t = img_to_array(img)
		#t = cvt(t)
		t = t.reshape((1,) + t.shape)
		x_train = np.concatenate((x_train, t))
print('the shape of x_train is:', x_train.shape)

#x_test
for i in range(1, 1591):
	img = load_img('blur/b' + num4(i) + '.png')
	if (i == 1):
		x_test = img_to_array(img)
		#x_test = cvt(x_test)
		x_test = x_test.reshape((1,) + x_test.shape)
	else:
		t = img_to_array(img)
		#t = cvt(t)
		t = t.reshape((1,) + t.shape)
		x_test = np.concatenate((x_test, t))
print('the shape of x_test is:', x_test.shape)

#y_train
for i in range(1, 2016):
	img = load_img('deblur/a' + num4(i) + '.png')
	if (i == 1):
		y_train = img_to_array(img)
		#y_train = cvt(y_train)
		y_train = y_train.reshape((1,) + y_train.shape)
	else:
		t = img_to_array(img)
		#t = cvt(t)
		t = t.reshape((1,) + t.shape)
		y_train = np.concatenate((y_train, t))
print('the shape of y_train is:', y_train.shape)

#y_test
for i in range(1, 1591):
	img = load_img('deblur/b' + num4(i) + '.png')
	if (i == 1):
		y_test = img_to_array(img)
		#y_test = cvt(y_test)
		y_test = y_test.reshape((1,) + y_test.shape)
	else:
		t = img_to_array(img)
		#t = cvt(t)
		t = t.reshape((1,) + t.shape)
		y_test = np.concatenate((y_test, t))
print('the shape of y_test is:', y_test.shape)

x_train /= 250
x_test /= 255
y_train /= 255
y_test /= 255

model = Sequential()
#layer1
model.add(Conv2D(filters = 128,
				 kernel_size = (19, 19),
				 activation = 'relu',
				 input_shape = (img_rows, img_cols, 3)))
#layer2
model.add(Conv2D(filters = 320,
				 kernel_size = (1, 1),
				 activation = 'relu'))
#layer3
model.add(Conv2D(filters = 320,
				 kernel_size = (1, 1),
				 activation = 'relu'))
#layer4
model.add(Conv2D(filters = 320,
				 kernel_size = (1, 1),
				 activation = 'relu'))
#layer5
model.add(Conv2D(filters = 128,
				 kernel_size = (1, 1),
				 activation = 'relu'))
#layer6
model.add(Conv2D(filters = 128,
				 kernel_size = (3, 3),
				 activation = 'relu'))
#layer7
model.add(Conv2D(filters = 512,
				 kernel_size = (1, 1),
				 activation = 'relu'))
#layer8
model.add(Conv2D(filters = 128,
				 kernel_size = (5, 5),
				 activation = 'relu'))
#layer9
model.add(Conv2D(filters = 128,
				 kernel_size = (5, 5),
				 activation = 'relu'))
#layer10
model.add(Conv2D(filters = 128,
				 kernel_size = (3, 3),
				 activation = 'relu'))
#layer11
model.add(Conv2D(filters = 128,
				 kernel_size = (5, 5),
				 activation = 'relu'))
#layer12
model.add(Conv2D(filters = 128,
				 kernel_size = (5, 5),
				 activation = 'relu'))
#layer13
model.add(Conv2D(filters = 256,
				 kernel_size = (1, 1),
				 activation = 'relu'))
#layer14
model.add(Conv2D(filters = 64,
				 kernel_size = (7, 7),
				 activation = 'relu'))
#layer15
model.add(Conv2D(filters = 3,
				 kernel_size = (7, 7)))

lr = 0.00004
decay = 0.8

sgd = optimizers.SGD(lr = lr, decay = decay)

model.compile(loss = 'mean_squared_error',
              optimizer = sgd)

model.summary()

batch_size = 4
epochs = 1
data_augmentation = 1

'''
model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = (x_test, y_test))
'''

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = (x_test, y_test))
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center = True,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip = False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size = batch_size),
                        steps_per_epoch = x_train.shape[0] // batch_size,
                        epochs = epochs,
						validation_data=(x_test, y_test))

save_model(model, 'deblur_lr' + str(lr) + '_epoch1.h5')

