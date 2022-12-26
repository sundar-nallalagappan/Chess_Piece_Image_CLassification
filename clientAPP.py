from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from inference import ImageClassification
from utils import decode_image

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.file_name = 'Input_image.jpg'
        self.classifier = ImageClassification(self.file_name)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_route():
    infer_image = request.json['image']
    decode_image(infer_image, clapp.file_name)
    result = clapp.classifier.predict_classes()
    return jsonify(result)

if __name__ == '__main__':
    clapp = ClientApp()
    app.run(host='0.0.0.0', port=8000, debug=True)
