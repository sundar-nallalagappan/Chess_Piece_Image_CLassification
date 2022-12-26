import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class ImageClassification:
    def __init__(self, file_name):
        self.file_name = file_name
        self.model = load_model('chess_piece_classifier.h5')

    def predict_classes(self):
        #model = load_model('chess_piece_classifier.h5')
        test_image = self.file_name
        print("predict classes:", test_image)

        test_image = image.load_img(test_image, target_size=(64,64))
        test_image = image.img_to_array(test_image)
        print(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        y_hat = self.model.predict(test_image)
        print(y_hat)

        y_hat_idx = np.argmax(y_hat)

        if y_hat_idx == 0:
            ans = 'Bishop'
        elif y_hat_idx == 1:
            ans = 'King'
        elif y_hat_idx == 2:
            ans = 'Knight'
        elif y_hat_idx == 3:
            ans = 'Pawn'
        elif y_hat_idx == 4:
            ans = 'Queen'
        elif y_hat_idx == 5:
            ans = 'Rook'
        else:
            ans = 'Uncertain'

        print(ans)

        return [{ "image" : ans}]
