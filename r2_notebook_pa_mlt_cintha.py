# -*- coding: utf-8 -*-
"""R2_notebook_PA_MLT_Cintha.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1p92Nz6w-R2Sbzl214i0z9sCFIu9IRV1j

**Nama** : Cintha Hafrida Putri

**Email** : devdan2510@gmail.com

**Id-Dicoding** : cintha_bang

#About Dataset

Dataset ini memiliki data 1K+ Peringkat dan Ulasan Produk Amazon sesuai detailnya yang tercantum di situs web resmi Amazon

**Fitur**
product_id - ID Produk

product_name - Nama Produk

category - Kategori Produk

discounted_price - Harga Diskon Produk

actual_price - Harga Asli Produk

discount_percentage - Persentase Diskon untuk Produk

rating - Rating Produk

rating_count - Jumlah orang yang memberikan rating di Amazon

about_product - Deskripsi Produk

user_id - ID pengguna yang menulis ulasan untuk Produk

user_name - Nama pengguna yang menulis ulasan untuk Produk

review_id - ID ulasan pengguna

review_title - Judul ulasan

review_content - Isi ulasan

img_link - Link Gambar Produk

product_link - Link Resmi Produk di Situs Web

img_link - Tautan Gambar Produk

product_link - Tautan Situs Web Resmi Produk
Inspirasi

Import library dan modul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

"""# Data Loading

Pada tahap ini saya memuat data yang akan dianalisis. Dimana saya mendownload terlebih dahulu dataset yang akan digunakan kemudian saya upload ke Google Drive agar lebih mudah untuk diakses.
"""

df = pd.read_csv('/content/drive/MyDrive/DICODING MACHINE LEARNING/Proyek pertama Machine Learning Terapan/amazon.csv')
df.head()

"""# Exploratory Data Analys (EDA)

Pada EDA ini saya akan mengecek struktur data, informasi data, mengidentivikasi adanya missing value, serta menampilkan distribusi kolom numerik.
"""

df.shape

df.info()

"""Mengidentifikasi Missing Value"""

# Identifikasi data yang hilang
missing_values = df.isnull().sum()
print("\nJumlah data yang hilang per kolom:")
print(missing_values)

"""Terdapat missing value pada kolom rating_count nantinya akan di handling saat Preprocessing

Tahap ini adalah visualisasi data numerik (namun saat ini typedatanya masih object)
"""

# Distribusi variabel numerik
numerical_cols = ['rating', 'rating_count', 'discount_percentage']
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols, 1):
  plt.subplot(1, 3, i)
  sns.histplot(df[col], kde=True)
  plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()

"""#Preprocessing

Dari proses EDA kita tahu bahwa beberapa data yang seharusnya memiliki tipe numerik sehingga membuat visualisasi kolom numerik tidak terbaca, serta terdapat missing value pada salah satu kolom.

Pada Tahap Preprocessing ini akan dilakukan penyesuaian tipe data, handling missing value, analisis statistik descriptif data, serta visualisasi kolerasi.
"""

# Konversi tipe data
df['discounted_price'] = df['discounted_price'].str.replace('₹','').str.replace(',','').astype(float)
df['actual_price'] = df['actual_price'].str.replace('₹','').str.replace(',','').astype(float)
df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = df['rating_count'].str.replace(',','').astype(float)

# Analisis statistik dasar
numeric_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']

# Statistik Deskriptif Variabel Numerik
print("\nStatistik Deskriptif Variabel Numerik:")
print(df[numeric_columns].describe())

"""Handling Missing Value

Mengisi missing value menggunakan median, karena kolom rating menggunakan skala 1-5 sehingga jika menggunakan mean bisa membuat data yang diisikan tidak umum (4.222). Kolom rating_count juga diisi dengan median karena memiliki distribusi skewed.
"""

# Rating
df['rating'].fillna(df['rating'].median(), inplace=True)

# Rating Count
df['rating_count'].fillna(df['rating_count'].median(), inplace=True)

# cek update data hilang
missing_values = df.isnull().sum()
print("\nJumlah data yang hilang per kolom:")
print(missing_values)

df.info()

"""Memisahkan kolom numerikal dan kategorikal untuk dapat dianalisis dengan mudah"""

# Memisahkan kolom numerikal dan kategorikal
numerical_columns = df.select_dtypes(include=['float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

print(f"Kolom numerikal: {list(numerical_columns)}")
print(f"Kolom kategorikal: {list(categorical_columns)}")

"""Pada tahap ini dilakukan visualisasi matriks korelasi dan distribusi kolom numerik"""

"""Visualisasi korelasi antar variabel numerik"""
numerical_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi antar Variabel Numerik')
plt.show()

# Visualisasi distribusi kolom numerik
numerical_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']

plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_columns, 1):
  plt.subplot(2, 3, i)
  sns.histplot(df[column], kde=True)
  plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()

"""Analisis outlier menggunakan box plots"""
numerical_columns = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']

plt.figure(figsize=(15, 5))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(1, 5, i)
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot {column}')
plt.tight_layout()
plt.show()

"""Outlier pada actual_price, discounted_price, rating_count adalah hal yang normal karena memang datanya pasti bervariasi. berbeda dengan discount_precentage dan rating yang datanya cenderung normal

# Modelling

## Model 1 K-Means

Model 1 K-Means menerapkan fitur selection dimana hanya beberapa fitur saja yang akan dianalisis
"""

# Select features for clustering
features = ['rating', 'rating_count', 'discount_percentage']
data_selected_features = df[features]

scaler = StandardScaler()
numeric_features = ['rating', 'rating_count', 'discount_percentage']  # Hanya fitur numerik yang relevan
data_selected_features[numeric_features] = scaler.fit_transform(data_selected_features[numeric_features])

# Setelah pembagian, lanjutkan dengan K-Means
inertias = []
silhouette_scores = []
ch_scores = []

K = range(2, 11)  # Jumlah cluster yang ingin diuji

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_selected_features)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_selected_features, kmeans.labels_))

silhouette_scores = []
for n_clusters in range(2, 11):  # Uji jumlah cluster dari 2 hingga 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_selected_features)
    silhouette_avg = silhouette_score(data_selected_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

# Plot silhouette scores untuk setiap jumlah cluster
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Numbers of Clusters")
plt.show()

# menentukan jumlah cluster yang optimal
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_selected_features)
    silhouette_avg = silhouette_score(data_selected_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

print(f"\nThe optimal number of clusters is: {optimal_n_clusters}")

kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(data_selected_features)

"""Model 1 K-Means menunjukkan nilai silhoutte score yang rendah sehingga perlu untuk mencoba skema lain

## Model 2 K-Means

Pada Model 2 K-Means saya menambahkan features baru untuk dianalisis yaitu price ratio dan review length
"""

df['price_ratio'] = df['actual_price'] / df['discounted_price']

df['review_length'] = df['review_content'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)

features = ['rating', 'rating_count', 'discount_percentage', 'price_ratio', 'review_length']

data_update_features = df[features]

# Setelah pembagian, lanjutkan dengan K-Means
inertias = []
silhouette_scores = []
ch_scores = []

K = range(2, 11)  # Jumlah cluster yang ingin diuji

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_update_features)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_update_features, kmeans.labels_))

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_update_features)
    silhouette_avg = silhouette_score(data_update_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

# Plot silhouette scores untuk setiap jumlah cluster
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Numbers of Clusters")
plt.show()

# menentukan jumlah cluster yang optimal
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_update_features)
    silhouette_avg = silhouette_score(data_update_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

print(f"\nThe optimal number of clusters is: {optimal_n_clusters}")

kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(data_selected_features)

"""Model 2 K-Means menunjukkan nilai silhoutte yang tinggi dengan cluster optimal 2"""

# Clustering dengan Jumlah Cluster Optimal
optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data_update_features)

# Evaluasi dengan Silhouette Score
silhouette_avg = silhouette_score(data_update_features, kmeans_labels)
print(f"Silhouette Score untuk {optimal_clusters} clusters: {silhouette_avg}")

# Menambahkan label cluster ke DataFrame asli
df['Cluster'] = kmeans_labels
print(df[['Cluster'] + features].head())

cluster_characteristics = df.groupby('Cluster')[features].describe()
print(cluster_characteristics)

"""## Visualisasi Hasil Clustering K-Means

Pada Tahap ini saya akan menampilkan visualisasi dari Model 2 sebagai model terbaik.
"""

cluster_0 =['Cluster'] == 0
cluster_1 =['Cluster'] == 1

for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Cluster', y=feature)
    plt.title(f'Distribusi {feature} berdasarkan Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.grid()
    plt.show()

"""## Model DBSCAN

Pada Tahap ini saya mencoba algoritma DBSCAN.
"""

# Pilih fitur yang akan digunakan untuk clustering
features = ['rating', 'rating_count', 'discount_percentage']
data_selected_features_dbscan = df[features]

# Penskalaan data
scaler = StandardScaler()
data_selected_features_dbscan = scaler.fit_transform(data_selected_features_dbscan)

# Membuat model DBSCAN dengan parameter yang dapat disesuaikan
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Melakukan clustering
dbscan_labels = dbscan.fit_predict(data_selected_features_dbscan)

# Menambahkan label cluster ke DataFrame asli
df['Cluster_DBSCAN'] = dbscan_labels

# Evaluasi model: Periksa jumlah kluster dan noise
n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_ = list(dbscan_labels).count(-1)

print(f'Estimated number of clusters: {n_clusters_}')
print(f'Estimated number of noise points: {n_noise_}')

# Menghitung metrik evaluasi jika ada lebih dari satu kluster
if n_clusters_ > 1:
    silhouette_avg = silhouette_score(data_selected_features_dbscan, dbscan_labels)

    print("\nDBSCAN Clustering Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette_avg}")
else:
    print("DBSCAN failed to form multiple clusters, consider adjusting eps or min_samples.")

# Menampilkan hasil clustering pada DataFrame
print(df[['Cluster_DBSCAN'] + features].head())

"""Score Silhouette DBSCAN ternyata bernilai lebih rendah daripada K-Means"""

# Reduksi dimensi menggunakan PCA untuk visualisasi 2D
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_selected_features_dbscan)

# Membuat scatter plot hasil klustering
plt.figure(figsize=(10, 7))
unique_labels = set(dbscan_labels)

# Warna untuk setiap kluster, termasuk noise (label -1)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Warna untuk noise
        col = [0, 0, 0, 1]

    class_member_mask = (dbscan_labels == k)

    xy = data_2d[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], label=f'Cluster {k}')

plt.title('DBSCAN Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

"""# Komparasi nilai Silhouette

Pada Tahap ini dilakukan komaparasi terkait ketiga Model yang telah dibuat
"""

# Calculate silhouette scores for each dataset
silhouette_data_update_features = silhouette_score(data_update_features, kmeans_labels)
silhouette_data_selected_features = silhouette_score(data_selected_features, kmeans.labels_)

# Assuming you have a DBSCAN model and its labels
silhouette_data_selected_features_dbscan = silhouette_score(data_selected_features_dbscan, dbscan_labels)


# Create a table or print the comparison
print("Silhouette Score Comparison:")
print(f"Model 1: {silhouette_data_selected_features}")
print(f"Model 2: {silhouette_data_update_features}")
print(f"Model 3: {silhouette_data_selected_features_dbscan}")

import matplotlib.pyplot as plt

# Data skor silhouette yang diperoleh
scores = {
    "Model 1 (KMeans)": silhouette_data_selected_features,
    "Model 2 (KMeans)": silhouette_data_update_features,
    "Model 3 (DBSCAN)": silhouette_data_selected_features_dbscan
}

# Ekstraksi nama model dan nilai skor untuk plot
labels = list(scores.keys())
values = list(scores.values())

# Membuat plot line
plt.figure(figsize=(10, 6))
plt.plot(labels, values, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title("Silhouette Score Comparison for Different Clustering Models and Feature Sets")
plt.xlabel("Dataset and Model")
plt.ylabel("Silhouette Score")
plt.ylim(0, 1)  # Skala silhouette score biasanya antara -1 hingga 1
plt.xticks(rotation=45, ha='right')
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Menambahkan nilai pada setiap marker
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=10)

# Menampilkan plot
plt.show()

# memutuskan skema terbaik
silhouette_scores = {
    "Model 1 (KMeans)": silhouette_data_selected_features,
    "Model 2 (KMeans)": silhouette_data_update_features,
    "Model 3 (DBSCAN)": silhouette_data_selected_features_dbscan
}

best_scheme = max(silhouette_scores, key=silhouette_scores.get)
best_score = silhouette_scores[best_scheme]

print("\nModel kluster terbaik:")
print(f"{best_scheme} dengan nilai silhoutte score {best_score:.4f}")
