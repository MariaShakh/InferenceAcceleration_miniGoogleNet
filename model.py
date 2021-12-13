from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras import backend as K
from tensorflow.keras.regularizers import l2, l1

from ttlayer.newttconv import *


A = 8
B = 8


def miniGoogleNet(width, height, depth, classes):
    def conv_module(x, K, kX, kY, A, B, stride, chanDim, padding="same"):
        # define convolution -> batch normalization -> ReLU
        if A <= 1 or B <= 1 or kX*kY == 1:
          # standart convolutional
          x = Conv2D(K, (kX, kY), strides=(stride[0], stride[1]), padding=padding)(x)
        else:
          # Tucker-2 +TT decomposition
          x = Conv2D(A, (1, 1), strides=(1, 1), padding=padding)(x)
          x = Conv2D(B, (kX, kY), strides=(1, 1), padding=padding)(x)
          x = ttconv(x, K, 2, window=(1, 1), strides=stride, padding=padding, regularizers=l2(0.01))
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        return x

    def inception_module(x, numK1x1, numK3x3, A, B, chanDim):
        # define two conv_modules, then concatenate them
        # across the channel dimension
        conv_1x1 = conv_module(x, numK1x1, 1, 1, -1, -1, [1, 1], chanDim)
        conv_3x3 = conv_module(x, numK3x3, 3, 3, A, B, [1, 1], chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
        return x

    def downsample_module(x, K, A, B, chanDim):
        # define the conv_module and pooling, then concatenate them
        # across the channel dimensions
        conv_3x3 = conv_module(x, K, 3, 3, A, B, [2, 2], chanDim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)
        return x

        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = conv_module(inputs, 96, 3, 3, A, B, [1, 1], chanDim)

        # two Inception modules followed by a downsample module
        x = inception_module(x, 32, 32, A, B, chanDim)
        x = inception_module(x, 32, 48, A, B, chanDim)
        x = downsample_module(x, 80, A, B, chanDim)

        # four Inception modules followed by a downsample module
        x = inception_module(x, 112, 48, A, B, chanDim)
        x = inception_module(x, 96, 64, A, B, chanDim)
        x = inception_module(x, 80, 80, A, B, chanDim)
        x = inception_module(x, 48, 96, A, B, chanDim)
        x = downsample_module(x, 96, A, B, chanDim)

        # two Inception modules followed by global POOL and dropout
        x = inception_module(x, 176, 160, A, B, chanDim)
        x = inception_module(x, 176, 160, A, B, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.2)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="miniGoogleNet")
        # return the constructed network architecture
        return model