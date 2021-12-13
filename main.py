import matplotlib.pyplot as plt
import time
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import miniGoogleNet


def lr_schedule(epoch):
    lr = 0.01
    if epoch > 90:
        lr = 0.0001
    elif epoch > 50:
        lr = 0.001
    elif epoch > 40:
        lr = 0.01
    print("lr = ", lr)
    return lr


# init
# initialize the initial learning rate, batch size, and number of epochs to train for
INIT_LR = lr_schedule(0)
BATCH_SIZE = 128
NUM_EPOCHS = 100

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

# load the CIFAR-10 dataset
print("[INFO] loading CIFAR-10 dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# scale the data to the range [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

print('Loading model...')
model = miniGoogleNet(32, 32, 3, len(labelNames))

# initialize the optimizer and compile the model
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# define the checkpoint
filepath = "/content/drive/My Drive/Colab/checkpoint_new.h5/"
lr_scheduler = LearningRateScheduler(lr_schedule)
# define the checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min', save_best_only=True, verbose=1)
callbacks_list = [checkpoint, lr_scheduler]

# start
start = time.time()
# train the network
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1,
    callbacks=callbacks_list)

duration = time.time() - start
print("{} s to get output".format(duration))

import pickle

with open('/content/drive/My Drive/Colab/new2tt', 'wb') as file_pi:
    pickle.dump(H.history, file_pi)

# evaluate the network
import time

print("[INFO] evaluating prediction...")
start = time.time()
predictions = model.predict(testX, batch_size=BATCH_SIZE)
duration = time.time() - start
print("{} s to get output".format(duration))
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames, digits=5))

# determine the number of epochs and then construct the plot title
N = np.arange(0, NUM_EPOCHS)
title = "Training Loss and Accuracy on CIFAR-10)"

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title(title)
plt.xlabel("Epoch №")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('/content/drive/My Drive/Colab/checkpoint_new.png')

model.summary()
