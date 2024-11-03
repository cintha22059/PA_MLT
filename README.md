# Laporan Proyek Machine Learning - Cintha Hafrida Putri

## Domain Proyek

Proyek ini berfokus pada analisis ulasan dan penilaian produk di Amazon 
untuk mendapatkan wawasan terkait preferensi pelanggan dan popularitas produk. 
Dataset ini mencakup lebih dari 1.000 data penilaian dan ulasan dengan berbagai atribut produk, 
seperti:

Informasi Produk: Product ID, Name, Category, Discounted Price, Actual Price, Discount Percentage.

Umpan Balik Pelanggan: Ratings, Number of Ratings, Review Title, and Content.

Informasi Pengguna: User ID and Name.

Tujuan utama dari proyek ini adalah memahami preferensi pelanggan, mengidentifikasi kategori produk yang populer, 
dan mengeksplorasi hubungan antara skor penilaian, tingkat diskon, serta fitur produk lainnya.

Analisis produk dan ulasan pelanggan di platform e-commerce seperti Amazon sangat 
penting karena dapat memberikan wawasan mengenai perilaku konsumen dan tren pasar.
Dalam lingkungan bisnis yang kompetitif, memahami kebutuhan dan preferensi pelanggan menjadi 
faktor kunci dalam meningkatkan kepuasan, memperkuat loyalitas, serta mendorong penjualan. 
Namun, karena data produk sering kali berjumlah besar dan kompleks, menemukan pola dan tren 
secara manual menjadi sangat sulit. Oleh karena itu, pendekatan berbasis machine learning,
seperti clustering, menjadi solusi efektif untuk mengatasi permasalahan ini. Clustering 
memungkinkan identifikasi kelompok produk atau pelanggan dengan karakteristik yang serupa,
sehingga strategi pemasaran dapat disesuaikan secara spesifik.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana menganalisis pola perilaku rating dan ulasan konsumen terhadap produk-produk Amazon?
- Apakah ada hubungan antara harga diskon, rating, dan jumlah ulasan terhadap performa produk?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Menganalisis dan memahami pola perilaku konsumen melalui rating dan ulasan untuk meningkatkan pengalaman berbelanja.
- Mengidentifikasi faktor-faktor yang mempengaruhi performa produk untuk optimasi strategi penetapan harga dan promosi
  
    ### Solution statements
    - Mengembangkan model clustering menggunakan algoritma K-Means dan DBSCAN untuk mengelompokkan produk berdasarkan karakteristik utama:
  Rating, Rating count, Discount percentage, Price ratio, Review length dan mengukur menggunakan metrik
evaluasi clustering silhoutte.

## Data Understanding
Dataset ini memiliki data 1465 Peringkat dan Ulasan Produk Amazon sesuai detailnya yang tercantum di situs web resmi Amazon
Jumlah baris : 1465
Jumlah Kolom : 16

### Variabel-variabel pada Amazon Sales dataset:
| Fitur                | Deskripsi                                         |
|----------------------|---------------------------------------------------|
| product_id           | ID Produk                                         |
| product_name         | Nama Produk                                       |
| category             | Kategori Produk                                   |
| discounted_price     | Harga Diskon Produk                               |
| actual_price         | Harga Asli Produk                                 |
| discount_percentage  | Persentase Diskon untuk Produk                    |
| rating               | Rating Produk                                     |
| rating_count         | Jumlah orang yang memberikan rating di Amazon     |
| about_product        | Deskripsi Produk                                  |
| user_id              | ID pengguna yang menulis ulasan untuk Produk      |
| user_name            | Nama pengguna yang menulis ulasan untuk Produk    |
| review_id            | ID ulasan pengguna                                |
| review_title         | Judul ulasan                                      |
| review_content       | Isi ulasan                                        |
| img_link             | Link Gambar Produk                                |
| product_link         | Link Resmi Produk di Situs Web                    |

Download datset : [https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/code](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/data)

## Exploratory Data Analysis
- Distribusi variabel numerik menunjukkan: Rating memiliki distribusi normal dengan median sekitar 4.1
Rating count memiliki distribusi skewed right
Discount percentage memiliki distribusi multimodal
- Analisis korelasi menunjukkan hubungan antara: Harga diskon dan harga asli (korelasi positif kuat)
Rating dan rating count (korelasi lemah)


## Data Preparation
Penanganan Tipe Data
- **Mengkonversi kolom harga**: Mengubah kolom harga dari string ke float dengan menghapus simbol mata uang (₹) dan tanda koma.
- **Mengubah kolom discount_percentage**: Mengubah dari string ke float dengan menghapus simbol '%'.
- **Konversi rating dan rating_count**: Mengubah tipe data kolom rating dan rating_count ke format numerik.

Penanganan Missing Value
- **Mengisi missing value pada kolom rating**: Mengisi dengan nilai median karena rating menggunakan skala 1-5.
- **Mengisi missing value pada kolom rating_count**: Mengisi dengan median karena distribusinya skewed.
- **Alasan**: Penggunaan median lebih tepat daripada mean untuk data yang tidak terdistribusi normal.

Feature Engineering
- **Fitur baru 'price_ratio'**: Membuat fitur 'price_ratio' yang dihitung dengan rumus `price_ratio = actual_price / discounted_price`.
- **Fitur 'review_length'**: Menambahkan fitur yang dihitung dari panjang kolom review_content.
- **Alasan**: Fitur ini dibuat untuk menangkap hubungan antara harga dan memberikan insight lebih detail tentang ulasan.

Standardisasi
- **Menggunakan StandardScaler**: Menormalkan fitur numerik agar skala antar fitur seragam.
- **Alasan**: Standardisasi ini penting agar tidak ada fitur yang mendominasi dalam proses clustering.

## Modeling

Algoritma yang Digunakan
1. **KMeans**
   - Model diinisialisasi dengan `fit` dan `fit_predict` pada data latih.
   - Digunakan untuk mengevaluasi jumlah cluster optimal berdasarkan Silhouette Score dan Elbow Method.

2. **DBSCAN**
   - Model dibuat dengan parameter yang dapat disesuaikan.
   - Evaluasi dilakukan untuk memeriksa jumlah cluster dan deteksi noise.

Parameter Model
- **KMeans**:
  - Pengaturan jumlah cluster (k) diuji dengan berbagai nilai untuk menemukan k optimal.
  - Evaluasi dilakukan dengan Silhouette Score.
- **DBSCAN**:
  - Parameter `eps` dan `min_samples` disesuaikan untuk mencapai clustering yang lebih baik pada data dengan distribusi non-linear.

Proses Improvement
- Model KMeans di-tuning dengan mencoba beberapa nilai `k` untuk mendapatkan score Silhouette yang tertinggi (~0.6 untuk k=3).
- DBSCAN digunakan untuk mengatasi noise dan data dengan bentuk yang lebih kompleks.

Alasan Pemilihan Model Terbaik
Model KMeans dipilih sebagai model terbaik karena mencapai skor Silhouette tertinggi dengan jumlah cluster optimal.

## Evaluation
Metrik Evaluasi untuk Model Clustering

Silhouette Score
- **Deskripsi**: Mengukur seberapa mirip objek dengan cluster-nya sendiri dibandingkan dengan cluster lain.
- **Range nilai**: -1 hingga 1.
- **Interpretasi**: Skor yang lebih tinggi menunjukkan cluster yang lebih baik.
- **Hasil**: Model mencapai skor tertinggi pada k=3 dengan nilai ~0.6.

Calinski-Harabasz Score
- **Deskripsi**: Mengukur rasio antara dispersi antar cluster dengan dispersi dalam cluster.
- **Interpretasi**: Skor yang lebih tinggi menunjukkan cluster yang lebih terpisah dengan baik.
- **Fungsi**: Memvalidasi hasil pemilihan jumlah cluster optimal.

Elbow Method
- **Deskripsi**: Metode untuk menentukan titik optimal di mana penambahan jumlah cluster tidak memberikan penurunan inertia yang signifikan.
- **Fungsi**: Membantu mengonfirmasi pemilihan jumlah cluster yang tepat.


**---Ini adalah bagian akhir laporan---**
