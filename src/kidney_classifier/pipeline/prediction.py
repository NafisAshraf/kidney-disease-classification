import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename, model_path=os.path.join("artifacts", "training", "model.h5")):
        self.filename = filename
        self.model_path = model_path

    def predict(self):
        model = load_model(self.model_path)

        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        probabilities = model.predict(test_image)
        result = np.argmax(probabilities, axis=1)
        confidence = float(np.max(probabilities))

        prediction = "Tumor" if result[0] == 1 else "Normal"
        return [{"image": prediction, "confidence": round(confidence, 4)}]

