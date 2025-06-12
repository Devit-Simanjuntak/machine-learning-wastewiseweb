# Proyek Machine Learning: Klasifikasi Gambar dengan CNN

Repositori ini berisi *notebook* dan model untuk proyek *machine learning* yang berfokus pada klasifikasi gambar menggunakan *Convolutional Neural Network* (CNN). Proyek ini digunakan untuk mengembangkan model klasifikasi gambar yang akan digunakan untuk [WasteWise Web](https://rizkyhanifaa.github.io/WasteWiseWeb ) untuk memberikan klasifikasi jenis sampah dari gambar yang diunggah atau diambil pengguna.

## Deskripsi Proyek

Tujuan dari proyek ini adalah untuk membangun, melatih, dan mengevaluasi model *deep learning* yang mampu mengklasifikasikan gambar ke dalam beberapa kategori yang telah ditentukan. Alur kerja proyek mencakup beberapa tahap utama:
1.  **Pengumpulan dan Eksplorasi Data**: Memuat dataset, melakukan analisis data eksplorasi (EDA) untuk memahami distribusi dan karakteristik gambar.
2.  **Pemodelan**: Membangun arsitektur model CNN, melatihnya pada data training, dan melakukan validasi.
3.  **Evaluasi**: Mengevaluasi performa model pada data tes dan menyimpan arsitektur model yang telah dilatih.

## Struktur File

Berikut adalah penjelasan singkat mengenai file-file penting dalam repositori ini:

* `Data_Collection_and_Exploration.ipynb`
    * *Notebook* ini berisi proses awal proyek. Tanggung jawab utamanya adalah memuat dataset, melakukan visualisasi data, dan analisis eksplorasi untuk mendapatkan wawasan dari data gambar.

* `ModellingKaggle_v2.ipynb`
    * Ini adalah *notebook* utama untuk proses pemodelan. Berisi kode untuk pra-pemrosesan data, augmentasi gambar, pembangunan arsitektur CNN, proses training, dan evaluasi performa model.
* `model_cnn/`
    * Direktori ini berisi file-file yang berkaitan dengan model yang telah dikonversi menggunakan TensorFlow.js dan disimpan.
    * **`model.json`**: File ini menyimpan arsitektur dari model CNN yang telah dibangun menggunakan Keras v3.8.0.

## Arsitektur Model

Model yang digunakan adalah **Sequential CNN** yang dirancang untuk tugas klasifikasi gambar. Arsitektur ini terdiri dari beberapa blok konvolusi yang diikuti oleh lapisan *fully-connected* untuk klasifikasi akhir. Berikut adalah rincian lapisan utamanya:

* **Input**: Model menerima input gambar dengan ukuran `224x224` piksel dan 3 channel warna (RGB).

* **Blok Konvolusi 1 & 2 (Feature Extraction Awal)**:
    * Dua lapis `Conv2D` dengan 32 filter dan kernel `(3,3)`.
    * Setiap lapisan `Conv2D` diikuti oleh `BatchNormalization` dan aktivasi `ReLU`.
    * Diakhiri dengan `MaxPooling2D` untuk mengurangi dimensi spasial dan `Dropout` (rate 0.25) untuk regularisasi.

* **Blok Konvolusi 3 & 4 (Feature Extraction Menengah)**:
    * Dua lapis `Conv2D` dengan 64 filter, diikuti `BatchNormalization` dan aktivasi `ReLU`.
    * Diakhiri `MaxPooling2D` dan `Dropout` (rate 0.25).

* **Blok Konvolusi 5 & 6 (Separable Convolution)**:
    * Menggunakan dua lapis `SeparableConv2D` dengan 128 filter untuk efisiensi komputasi, masing-masing diikuti `BatchNormalization` dan aktivasi `ReLU`.
    * Diakhiri `MaxPooling2D` dan `Dropout` (rate 0.25).

* **Blok Konvolusi 7 & 8 (Deep Feature Extraction)**:
    * Dua lapis `SeparableConv2D` dengan 256 filter, `BatchNormalization` dan `ReLU`.
    * Diikuti `MaxPooling2D` dan `Dropout` (rate 0.3).

* **Blok Konvolusi 9**:
    * Satu lapis `Conv2D` dengan 512 filter, `BatchNormalization` dan `ReLU`.
    * Diikuti `MaxPooling2D` dan `Dropout` (rate 0.3).

* **Lapisan Klasifikasi (Classifier Head)**:
    * `GlobalAveragePooling2D` untuk meratakan *feature map*.
    * Tiga lapisan `Dense` (fully-connected) dengan 512, 256, dan 128 neuron secara berurutan. Setiap lapisan menggunakan `BatchNormalization`, aktivasi `ReLU`, dan `Dropout` untuk regularisasi.
    * **Lapisan Output**: Sebuah lapisan `Dense` akhir dengan **12 neuron** dan aktivasi **`softmax`**, yang menunjukkan bahwa model ini melakukan klasifikasi untuk **12 kelas yang berbeda**.

## Teknologi yang Digunakan

* **Python 3**
* **Jupyter Notebook**
* **Library Utama**:
    * `TensorFlow` / `Keras`
    * `Scikit-learn`
    * `Pandas`
    * `NumPy`
    * `Matplotlib`
    * `Seaborn`

## Instalasi

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**
    ```bash
    git clone [<URL_REPOSITORI_ANDA>](https://github.com/Devit-Simanjuntak/machine-learning-wastewiseweb).git
    cd MachineLearning
    ```

2.  **Buat dan aktifkan lingkungan virtual (disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal semua dependensi yang dibutuhkan:**
    ```bash
    pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
    ```

## Cara Menjalankan

1.  **Unduh Dataset**
    * **[Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)**
    * **[Garbage Classification Dataset Balanced](https://drive.google.com/file/d/1jqN5HK-9S2mLkSllhPjc6uQ1ijg7Lcc_/view?usp=drive_link)**

2.  **Jalankan Notebook secara Berurutan:**
    * Buka Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
    * Jalankan `Data_Collection_and_Exploration.ipynb` terlebih dahulu untuk memahami data.
    * Selanjutnya, jalankan `ModellingKaggle_v2.ipynb` untuk melatih model Anda dari awal. Pastikan path dataset sudah benar.

## Hasil

Model CNN yang dikembangkan berhasil mencapai metrik performa sebagai berikut (Anda bisa mengisi bagian ini dengan hasil spesifik Anda):

* **Akurasi Training**: `85.80%`
* **Akurasi Validasi**: `87.20%`

Detail lebih lanjut mengenai matriks evaluasi seperti *confusion matrix* atau kurva *loss* dapat ditemukan di dalam `ModellingKaggle_v2.ipynb`.
