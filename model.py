#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from glob import glob

#resize all images to the expected size
image_size = [224, 224]

#set train, test, validation dataset path
train_path = '/content/drive/My Drive/Datasets/Malaria detection big data/cell_images'

base_model = keras.applications.vgg19.VGG19(input_shape=image_size + [3], weights = "imagenet", include_top=False)

for layer in base_model.layers:
  layer.trainable = False

out_classes = glob('/content/drive/My Drive/Datasets/Malaria detection big data/cell_images/*')

layer_flatten = keras.layers.Flatten()(base_model.output)
output = keras.layers.Dense(len(out_classes), activation="softmax")(layer_flatten)
model = keras.Model(inputs=base_model.input, outputs=output)

print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.2)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 subset='training')

test_set = train_datagen.flow_from_directory(train_path,
                                             target_size = (224, 224),
                                             batch_size = 16,
                                             class_mode = 'categorical',
                                             subset='validation')

#train the model
history = model.fit_generator(
  training_set,
  validation_data = test_set,
  epochs = 5,
  steps_per_epoch = len(training_set),
  validation_steps = len(test_set)
)

#save the model as an h5 file
model.save('/content/drive/My Drive/Datasets/Malaria detection big data/model_vgg19.h5')

#plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

print(model.evaluate(test_set))

#using the model to make predictions
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)