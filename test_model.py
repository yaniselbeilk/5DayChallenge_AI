import tensorflow as tf
import numpy as np
import pathlib

img_height = 180
img_width = 180


#Load les données
data_dir = pathlib.Path("ImageFound/")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#Paramèttres
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names

import pathlib
#Load les données
model = tf.keras.models.load_model("image_classifier.h5")
data_dir = pathlib.Path("ImageFound/microsoft/Microsoft (1).jpg")
img = tf.keras.utils.load_img(
    data_dir, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)