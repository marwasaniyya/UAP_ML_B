# ✨ Analisis Sentimen Ulasan Pengguna Aplikasi Gojek: Perspektif Pengalaman Pengguna di Indonesia ✨

## 📑 Table of Contents
1. [📋 Deskripsi Proyek](#deskripsi-proyek)
2. [⚙️ Langkah Instalasi](#langkah-instalasi)
3. [🗂️ Overview Dataset](#overview-dataset)
4. [🧠 Deskripsi Model](#deskripsi-model)
   - [📈 LSTM](#lstm)
   - [📊 BERT](#bert)
5. [📊 Hasil dan Analisis](#hasil-dan-analisis)
6. [📊 Kesimpulan perbandingan model LSTM dan BERT](#Kesimpulan-perbandingan-model-LSTM-dan-BERT)
7. [🔗 Link Live Demo](#link-live-demo)
8. [👨‍💻 Author](#author)

---

## 📋 Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan model analisis sentimen pada ulasan pengguna aplikasi Gojek, dengan memanfaatkan rating bintang 1 hingga 5 sebagai indikator sentimen. Pendekatan ini dirancang untuk memahami perspektif pengalaman pengguna di Indonesia dan memberikan wawasan berharga guna meningkatkan kualitas layanan aplikasi.  

Analisis ini mengelompokkan ulasan berdasarkan rating bintang, di mana bintang 1 dan 2 mencerminkan sentimen negatif, bintang 3 dianggap netral, dan bintang 4 serta 5 mencerminkan sentimen positif. Dengan model ini, ulasan pengguna dapat diolah menjadi informasi strategis untuk mendukung pengambilan keputusan dan perbaikan layanan.

---

## ⚙️ Langkah Instalasi
Ikuti langkah-langkah berikut untuk menginstal dependencies dan menjalankan aplikasi:

1. **Clone Repository:**
   ```bash
   git init
   git add .
   git commit -m "Inisialisasi proyek"
   git remote add origin https://github.com/marwasaniyya/UAP.git
   git branch -M main
   git push -u origin main

   commit
   git status
   git add (sesuai file yang ditambahkan)
   git commit -m "coba"
   git push origin main
  
   ```

2. **Buat Virtual Environment:**
   ```bash
   python -m venv env
   env\Scripts\activate   # Untuk Windows
   ```

3. **Instal Dependencies:**
   ```bash
   pip install pdm
   pdm init
   pdm add streamlit
   pdm add tensorflow
   pdm add joblib
   pdm add scikit-learn
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Web:**
   ```bash
   streamlit run app.py
   ```

---
## 🗂️ Overview Dataset
Dataset diambil dari [Master Data File Full Columns](https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia). Dataset ini mencakup:
- **Periode Pengumpulan Data:** 05 November 2021 hingga 02 Februari 2024
- **Jumlah Variabel:** 5 kolom
- **Jumlah Observasi:** 25.011 baris
  Data dibagi menjadi dua bagian: 80% untuk Training Set dan 20% untuk Testing Set. Dimana pada setiap Set, terdapat 5 Label Class yaitu:
  - Label "0" menunjukkan bahwa berita memiliki rating 1, yang merupakan rating terjelek.
  - Label "1" menunjukkan bahwa berita memiliki rating 2, yang sedikit lebih baik dibandingkan rating 1.
  - Label "2" menunjukkan bahwa berita memiliki rating 3, yang merupakan rating sedang atau netral.
  - Label "3" menunjukkan bahwa berita memiliki rating 4, yang cukup bagus.
  - Label "4" menunjukkan bahwa berita memiliki rating 5, yang merupakan rating terbaik.
    
## 🧠 Deskripsi Model

### 📈 LSTM
Model LSTM (Long Short-Term Memory) digunakan untuk menangkap konteks temporal dalam teks. Model ini dirancang untuk memahami hubungan antar kata dalam urutan teks sehingga dapat meningkatkan akurasi klasifikasi sentimen.

#### 🧠 Struktur LSTM 
Model LSTM memiliki arsitektur khusus yang dirancang untuk menangani masalah vanishing gradient dalam jaringan saraf berulang (RNN). Struktur utamanya terdiri dari tiga gerbang utama:  Forget Gate: Memutuskan informasi mana yang akan dilupakan dari memori. Input Gate: Memutuskan informasi baru mana yang akan ditambahkan ke memori. Output Gate: Memutuskan bagian dari memori yang akan dikeluarkan sebagai output.

Berikut adalah ilustrasi struktur LSTM: 

![image](https://github.com/user-attachments/assets/c0801fac-bf4c-417a-ba27-28d797fabfae)

#### 🔧Prepocessing 

Pada tahap preprocessing  mencakup data cleaning untuk menghilangkan noise atau data yang tidak relevan, tokenisasi untuk memecah teks menjadi kata atau token, dan stopword removal untuk menghapus kata-kata yang tidak penting seperti "dan", "atau", dan "adalah". dilanjut dengan melakukan splitting dataset menjadi 2 (Training, dan Testing) sesuai dengan penjelasan pada Dataset.

Hasil dari model lstm yang telah di bangun

| Layer Type               | Output Shape          | Number of Parameters |
|--------------------------|-----------------------|----------------------|
| Embedding                | (None, 100, 256)      | 1,280,000            |
| LSTM                     | (None, 256)           | 525,312              |
| Dense                    | (None, 5)             | 1,285                |



#### 📊 Hasil Evaluasi Model LSTM
   1. **Training Accuracy**: Secara bertahap meningkat hingga mencapai sekitar 95%, menunjukkan bahwa model berhasil mempelajari pola pada data pelatihan dengan sangat baik.
   2. **Validation Accuracy**: Stabil di kisaran 75%, yang mencerminkan kemampuan model untuk menggeneralisasi data validasi dengan baik meskipun tidak setinggi data pelatihan.
   3. **Testing Accuracy**: Lebih rendah di kisaran 70%, menunjukkan adanya kesulitan model dalam menggeneralisasi pada data yang lebih beragam atau berbeda dari data pelatihan.

#### 📝 Classification Report LSTM

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.69      | 0.74   | 0.71     | 1445    |
| 1     | 0.15      | 0.13   | 0.14     | 248     |
| 2     | 0.10      | 0.09   | 0.09     | 241     |
| 3     | 0.09      | 0.05   | 0.06     | 283     |
| 4     | 0.87      | 0.90   | 0.88     | 2785    |

Accuracy: 73%

#### 📉 Grafik Akurasi LSTM

![image](![download1](https://github.com/user-attachments/assets/1b3ee0cc-ffa3-4008-926a-78e3c2a51368)

Training accuracy meningkat secara bertahap hingga mencapai sekitar 95% setelah beberapa epoch. Testing accuracy stabil di kisaran 75%, mengindikasikan adanya overfitting ringan. 

#### 📉 Grafik Loss LSTM

![image]![download2](https://github.com/user-attachments/assets/2488c9ab-c921-4d3a-b0d1-a8ca21fdc904)

Training loss menurun secara konsisten selama pelatihan. Namun, testing loss mulai meningkat setelah beberapa epoch, yang merupakan tanda bahwa model mengalami overfitting. 

#### 🧩 Confusion Matrix 📊
Hasil dari Confusion Matrix

![image]![download3](https://github.com/user-attachments/assets/a2c37592-bd72-4d0c-94b4-3d6c20f71f9f)

----

### 📊 BERT
Model BERT (Bidirectional Encoder Representations from Transformers) digunakan untuk memanfaatkan representasi teks berbasis transformer yang lebih kaya. Model ini sangat efektif dalam memahami konteks dua arah dalam teks.

#### 🧠 Struktur LSTM 
Arsitektur BERT terdiri dari tiga bagian utama: input layer yang menggabungkan token, segment, dan position embeddings; encoder berbasis Transformer dengan self-attention untuk pemrosesan bidirectional; dan output layer yang menghasilkan representasi vektor untuk tugas lanjutan seperti klasifikasi atau NER.

Berikut adalah ilustrasi struktur BERT: 

![image](https://github.com/user-attachments/assets/583004e0-9546-47c4-bb65-26435c08936d)


#### 🔧Prepocessing 

Pada tahap preprocessing  mencakup data cleaning untuk menghilangkan noise atau data yang tidak relevan, tokenisasi untuk memecah teks menjadi kata atau token, dan stopword removal untuk menghapus kata-kata yang tidak penting seperti "dan", "atau", dan "adalah". dilanjut dengan melakukan splitting dataset menjadi 2 (Training, dan Testing) sesuai dengan penjelasan pada Dataset.

#### 📊 Hasil Evaluasi Model BERT
1. **Training Accuracy**: Mengalami peningkatan signifikan hingga mencapai 81%.
2. **Validation Accuracy**: Stabil di sekitar 79%, menunjukkan kemampuan model untuk menggeneralisasi data yang belum terlihat.
3. **Testing Accuracy**: Sekitar 78%, menandakan bahwa model memiliki kinerja yang baik dalam melakukan prediksi pada data baru setelah pelatihan.

#### 📝 Classification Report BERT
| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.00      | 0.00   | 0.00     | 0       |
| 1     | 0.00      | 0.00   | 0.00     | 241     |
| 2     | 0.29      | 0.05   | 0.08     | 220     |
| 3     | 0.50      | 0.01   | 0.01     | 300     |
| 4     | 0.90      | 0.93   | 0.91     | 2787    |
Accuracy : 73%

#### 📉 Grafik BERT

1. **Accuracy:**
   
![image]![download4](https://github.com/user-attachments/assets/1569ed33-bff4-4e8b-bdda-7bd4ccd5ccc3)

Training accuracy meningkat secara bertahap dan mencapai sekitar 81% setelah beberapa epoch. Validation accuracy stabil di kisaran 79%, menunjukkan bahwa model mampu mempertahankan generalisasi yang baik.

3. **Loss:**
   
![image]![download5](https://github.com/user-attachments/assets/e17bb6ab-78eb-4a98-ad7c-ff47d6c182a6)

Training loss terus menurun secara signifikan, menunjukkan bahwa model berhasil mempelajari data pelatihan dengan baik. Namun, validation loss cenderung meningkat setelah beberapa epoch, menandakan adanya potensi overfitting pada model.

#### 🧩 Confusion Matrix 📊
Hasil dari Confusion Matrix

![image]![download6](https://github.com/user-attachments/assets/8f57a8a4-7521-41e9-a010-d79466fa9a1d)

## 📊 Hasil dan Analisis

### 📈 Perbandingan Performa Model
| Model | Training Accuracy | Validation Accuracy | Testing Accuracy |
|-------|-------------------|---------------------|------------------|
| LSTM  | 95%               | 75%                 | 70%              |
| BERT  | 81%               | 79%                 | 78%              |

---

### 📊 Kesimpulan perbandingan model LSTM dan BERT

1. **Akurasi** :
   - *LSTM* :
     * Akurasi training mencapai 95% dan akurasi test mencapai 70%.
     * menunjukkan bahwa model berhasil mempelajari pola pada data pelatihan dengan sangat baik.
   - *BERT*:
     * Akurasi training meningkat menjadi 81% dan akurasi test mencapai 78%.
     * Grafik menunjukkan peningkatan akurasi secara signifikan dari awal hingga akhir epoch. Hal ini menunjukkan bahwa model berhasil mempelajari pola dari data pelatihan dengan baik.generalisasi yang lebih kuat.

2. **Loss**:
   - *LSTM*
      * meningkat setelah beberapa epoch, yang menandakan model mengalami overfitting.  
   - *BERT*
     * Loss pada data pelatihan terus menurun secara konsisten seiring bertambahnya epoch. Ini menandakan bahwa model semakin baik dalam meminimalkan kesalahan selama pelatihan. 
 
 3. **Kemampuan generalisasi**:
    - *LSTM*
       * Dapat memahami urutan teks dengan baik, namun cenderung overfit.
    - *BERT*
       * Model dengan  arsitektur transformator dua arah ini lebih baik dalam memahami hubungan yang kompleks dan menunjukkan kemampuan generalisasi yang lebih baik.

4. **Fleksibilitas penyesuaian**:
   - *LSTM*
      * Cocok untuk data dengan urutan kronologis yang jelas, namun terbatas dalam memahami hubungan nonlinier yang kompleks.
   - *BERT*
      * Lebih baik dalam menangani konteks kompleks dan hubungan non-linier, memungkinkan dapat melakukan tugas analisis sentimen terperinci dengan lebih efektif.

### **Kesimpulan akhir**:
Dari tabel perbandingan performa model, dapat dilihat bahwa LSTM menunjukkan akurasi pelatihan yang tinggi (95%), tetapi mengalami penurunan signifikan pada akurasi validasi (75%) dan pengujian (70%), yang mengindikasikan kemungkinan overfitting. Sementara itu, BERT memiliki akurasi pelatihan yang lebih rendah (81%) tetapi menunjukkan akurasi yang lebih konsisten pada data validasi (79%) dan pengujian (78%). Hal ini menunjukkan bahwa BERT lebih mampu generalisasi pada data yang tidak terlihat, menjadikannya pilihan yang lebih baik untuk tugas klasifikasi teks dalam konteks ini.

---

## 🔗 Link Live Demo
Aplikasi web telah di-deploy dan dapat diakses melalui tautan berikut:  
[Live Demo Aplikasi](https://UAP.streamlit.app/)


---

## 👨‍💻 Author  
👨‍💻 [Saniyya Ruzzy Marwa](https://github.com/marwasaniyya)
