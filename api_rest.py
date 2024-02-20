from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pathlib
import io
import base64

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/find", methods=['POST'])
def hello_world():
    # Paramèttres
    batch_size = 32
    img_height = 180
    img_width = 180

    # Load les classes
    data_dir = pathlib.Path("ImageFound/")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    class_names = train_ds.class_names

    # Récupération de l'image
    url = request.json.get('url')
    b = url.split("base64,")[1]
    img_data = io.BytesIO(base64.b64decode(b))

    # Load les données
    model = tf.keras.models.load_model("image_classifier.h5")
    #img_data = tf.keras.utils.get_file("img", origin=url)
    img = tf.keras.utils.load_img(
        img_data, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return {"name":format(class_names[np.argmax(score)]), "score":round(100 * np.max(score),2)}

if __name__ == '__main__':
    app.run(debug=True)