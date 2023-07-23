import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

# Fotoğrafların ve kategorilerin tutulacağı listeleri oluştur
images = []
categories = []

# Kategoriye ait klasör yolunu oluştur
folder_path = 'Veri Seti'  # Klasör yolunu düzenle
folders = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

# Kategori klasöründeki tüm resimleri al ve işle
for folder in folders:
    folder_path = os.path.join('Veri Seti', folder)
    image_names = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        image = image.resize((128, 128))  # Boyutlandırma işlemini isteğe bağlı olarak yapabilirsiniz
        image = np.array(image) / 255.0  # Normalizasyon
        images.append(image)
        categories.append(folder)

# Giriş ve çıkış verilerini numpy dizilerine dönüştür
X = np.array(images)
categories = np.array(categories)

# Kategorileri sayısal değerlere dönüştür
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(categories)

# Eğitim veri setini ve hedef değerleri oluştur
train_X = X
train_y = train_y_encoded

# Resimleri rastgele görüntüle
random_images = []
for i in range(len(folders)):
    category_indices = np.where(train_y == i)[0]
    random_image_index = random.choice(category_indices)
    random_images.append(random_image_index)

# Resimleri görüntüleme
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(X[random_images[i]])
    ax.axis('off')
    ax.set_title(label_encoder.inverse_transform([train_y[random_images[i]]])[0])
plt.tight_layout()
plt.tight_layout()
plt.show()

# Renk skalası çıkarma
mean_colors = []
for image_index in random_images:
    mean_color = np.mean(X[image_index], axis=(0, 1))
    mean_colors.append(mean_color)

mean_colors = np.array(mean_colors)

# Özellik haritalarını oluşturma
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(mean_colors[i].reshape(1, 1, 3))
    ax.axis('off')
    ax.set_title(label_encoder.inverse_transform([train_y[random_images[i]]])[0])
plt.tight_layout()
plt.show()

# Data Augmentation uygula
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(train_X)

# Önceden eğitilmiş bir model kullanarak transfer öğrenme yap
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output

# GlobalAveragePooling2D katmanı ekle
x = GlobalAveragePooling2D()(x)

# Yoğun dense katmanı ekle
x = Dense(128, activation='relu')(x)
x = Dropout(0.15)(x)

predictions = Dense(len(folders), activation='softmax')(x)  # Kategori sayısına uygun çıkış katmanı boyutu ve aktivasyon fonksiyonu
model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derle ve eğit
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(datagen.flow(train_X, train_y, batch_size=16), epochs=21)

# Modeli kaydet
model.save('yemek_tanıma_yapay_zeka_modeli.hdf5')
