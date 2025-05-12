import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === Ayarlar ===
saved_model_dir = r'C:\Users\sbber\Desktop\plant_diase\saved_model'
classes_path = r'C:\Users\sbber\Desktop\plant_diase\classes.txt'

Tk().withdraw()
img_path = askopenfilename(title="Bir Görsel Seçin", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

if not img_path:
    print("Görsel seçilmedi.")
    sys.exit(1)

if not os.path.exists(saved_model_dir):
    print(f"Model klasörü bulunamadı: {saved_model_dir}")
    sys.exit(1)

try:
    model = tf.saved_model.load(saved_model_dir)
    print("Model başarıyla yüklendi (tf.saved_model.load).")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    sys.exit(1)

infer = model.signatures['serving_default']
print(f"Model signature'ları: {model.signatures}")
for key in infer.structured_outputs:
    print(f"Output key: {key}")

if not os.path.exists(classes_path):
    print(f"Sınıf dosyası bulunamadı: {classes_path}")
    sys.exit(1)

with open(classes_path, 'r') as f:
    class_names = [line.strip() for line in f if line.strip()]

img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

try:
    predictions = infer(tf.convert_to_tensor(img_array))
    output_key = list(predictions.keys())[0]
    predicted_class = class_names[np.argmax(predictions[output_key].numpy())]
except Exception as e:
    print(f"Tahmin hatası: {e}")
    sys.exit(1)

print(f"Tahmin edilen sınıf: {predicted_class}")
plt.imshow(img)
plt.title(f"Tahmin: {predicted_class}")
plt.axis('off')
plt.show()
