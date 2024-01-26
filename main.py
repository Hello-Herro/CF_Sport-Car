import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("Sport car price.csv")

# Menggabungkan kolom 1 dan kolom 2
df['Car'] = df['Car Make'].astype(str) + " " + df['Car Model'].astype(str)

# Menghapus kolom 1 dan kolom 2
df = df.drop(['Car Make', 'Car Model'], axis=1)

# Menentukan posisi kolom baru (kolom_gabungan) di DataFrame
posisi_kolom_baru = 0  # Misalnya, posisi pertama

# Memindahkan kolom baru ke posisi yang diinginkan
df.insert(posisi_kolom_baru, 'Car', df.pop('Car'))

# Memilih fitur yang akan digunakan untuk pengelompokan
features = df[['Engine Size (L)', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)', 'Price (in USD)']]

# Mengonversi nilai-nilai fitur ke dalam bentuk teks
features = features.astype(str).apply(' '.join, axis=1)

# Membangun matriks TF-IDF dari teks fitur
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(features)

# Menghitung kesamaan kosinus antar mobil
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi berdasarkan kesamaan kosinus
def get_recommendations(car_name, cosine_similarities, df):
    car_index = df.index[df['Car'] == car_name].tolist()[0]
    similar_cars = list(enumerate(cosine_similarities[car_index]))
    similar_cars = sorted(similar_cars, key=lambda x: x[1], reverse=True)
    similar_cars = similar_cars[1:16]  # Mengambil 5 mobil teratas (tidak termasuk dirinya sendiri)
    recommended_cars = [df.iloc[i[0]]['Car'] for i in similar_cars]
    return recommended_cars

# Contoh penggunaan untuk mendapatkan rekomendasi mobil
car_name = 'Lamborghini Huracan'
recommendations = get_recommendations(car_name, cosine_similarities, df)

# Menampilkan hasil rekomendasi
print(f'Rekomendasi mobil untuk {car_name}:')
print(recommendations)