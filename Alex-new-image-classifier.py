import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
model = load_model(r"E:\NNDL CNN\ai_vs_real_art_model.h5")
IMAGE_SIZE = (224, 224)
class_labels = ['AI Art', 'Real Art']
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    prediction = model.predict(img_array)[0][0]
    predicted_class = class_labels[int(round(prediction))]
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({prediction:.2f})")
    plt.axis('off')
    plt.show()
    print(f"Prediction Score: {prediction:.4f} -> Classified as: {predicted_class}")
img_path = r"C:\Users\amith\Downloads\image-1.jpg"  # 🔁 Replace this with your test image path
predict_image(img_path)
