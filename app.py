from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Cargar el modelo
model = load_model('mnt/data/my_fruit_classifier_model_mobilenetv2.h5')

# Definir las clases de frutas según el notebook de Kaggle
classes = [
    'Apple', 'Banana', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Carambula', 'Cherry', 'Chestnut',
    'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Grape Blue',
    'Grape Pink', 'Grape White', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki',
    'Kiwi', 'Kumquats', 'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mangostan', 'Melon Piel de Sapo',
    'Mulberry', 'Nectarine', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat', 'Pear', 'Pear Monster',
    'Pear Williams', 'Pepino', 'Physalis', 'Pineapple', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Potato Red',
    'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Tamarillo', 'Tangelo',
    'Tomato', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Walnut', 'Watermelon'
]

def prepare_image(image):
    image = image.resize((224, 224))  # Change size to 224x224
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image = Image.open(image_file)
            image = prepare_image(image)
            prediction = model.predict(image)
            print(prediction)
            predicted_class_index = np.argmax(prediction, axis=-1)[0]
            predicted_class = classes[predicted_class_index]

            return render_template('result.html', prediction=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
