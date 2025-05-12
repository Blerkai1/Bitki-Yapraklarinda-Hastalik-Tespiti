import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import os

# Eğitim parametreleri
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Kaggle'den indirilen verinin dizini
DATA_DIR = "plant_diase\dataset" 

# Eğitim ve sonuç verileri (boş başlatılıyor, doldurulacak)
history_data = {
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": []
}

# Model eğitimi
def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    # Modeli kaydetme ve yazdirma
    model.save("saved_model.pb", save_format='pb')
    print("Model kaydedildi: saved_model.pb")  

    
    history_data["accuracy"] = history.history["accuracy"]
    history_data["val_accuracy"] = history.history["val_accuracy"]
    history_data["loss"] = history.history["loss"]
    history_data["val_loss"] = history.history["val_loss"]

# Grafik çizimi ve tkinter arayüz oluşturulması
def show_graph():
    button.grid_forget()

    train_model()

    epochs = list(range(1, len(history_data["accuracy"]) + 1))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data["accuracy"], label='Doğruluk (Training)', color='blue')
    plt.plot(epochs, history_data["val_accuracy"], label='Doğruluk (Validation)', color='green')
    plt.title('Doğruluk Grafiği')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.ylim([0.4, 1.0])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_data["loss"], label='Kayıp (Training)', color='red')
    plt.plot(epochs, history_data["val_loss"], label='Kayıp (Validation)', color='orange')
    plt.title('Kayıp Grafiği')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.ylim([0.0, 1.2])
    plt.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = ImageTk.PhotoImage(img)

    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=1, column=0)
    buf.close()

# arayüzü başlatma kısmı
root = tk.Tk()
root.title("Model Eğitim Sonuçları")

button = tk.Button(root, text="Eğitim Sonuçlarını Göster", command=show_graph)
button.grid(row=0, column=0)

root.mainloop()
