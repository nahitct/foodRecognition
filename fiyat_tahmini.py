import random
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
# 1. Excel dosyasından fiyatlar ve kategori isimlerini çek
data = pd.read_excel('Yemek_fiyatlari.xlsx')  # Excel dosyasının yolu
categories = data['kategori'].values
prices = data['fiyat'].values

# 2. 'yemek_tanıma_yapay_zeka_modeli.hdf5' modülünü yükle
model = load_model('yemek_tanıma_yapay_zeka_modeli.hdf5')  # Modelin yolunu düzenleyin

# 3-5. Doğrulama Veri Seti klasörüne girip rastgele kategoriden rastgele bir resim seçip tahmin yap
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
total_price = 0

for i, ax in enumerate(axs.flatten()):
    # Rastgele bir kategori seç
    random_category = random.choice(categories)

    # Doğrulama Veri Seti klasöründeki rastgele bir resim seç
    folder_path = os.path.join('Doğrulama Veri Seti', random_category)
    image_names = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    random_image_name = random.choice(image_names)
    random_image_path = os.path.join(folder_path, random_image_name)

    # Seçilen resmi görüntüle
    image = Image.open(random_image_path)
    ax.imshow(image)
    ax.axis('off')
    
    # Tahmin yap
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_category_index = np.argmax(prediction)
    predicted_category = categories[predicted_category_index]
    predicted_price = prices[predicted_category_index]
    
    # Tahmin sonucunu ve fiyatı ekrana yazdır
    ax.set_title(f"Yemek Kategorisi: {predicted_category}\nFiyat: {predicted_price} TL")

    # Toplam fiyatı güncelle
    total_price += predicted_price

# 5. Bu dört resmin üstlerinde kategori tahminleri ve fiyatlarıyla yazılı şekilde bir resim çıktısı ver
plt.tight_layout()


# 6. Bu dört resmin en alt kısmında da yaptığın tahminlere göre toplam fiyat yazsın
plt.figtext(0.5, 0.01, f"Toplam Fiyat: {total_price} TL", ha="center", fontsize=12, bbox={"facecolor":"gray", "alpha":0.5, "pad":5})
plt.tight_layout()
plt.show()