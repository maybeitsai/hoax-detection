# Submission 1: Hoax Detection
Nama: Harry Mardika

Username dicoding: hkacode

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Indonesia False News (Hoax) Dataset](https://www.kaggle.com/datasets/muhammadghazimuharam/indonesiafalsenews/data) |
| Masalah | Dataset ini berisi teks berita dalam bahasa Indonesia yang perlu diklasifikasikan apakah berita tersebut hoaks atau tidak. Tantangan utama adalah mengembangkan model yang dapat membedakan dengan tepat antara berita hoaks dan berita asli. |
| Solusi machine learning | Menggunakan teknik machine learning untuk membangun model klasifikasi teks yang dapat memprediksi keaslian sebuah berita berdasarkan narasinya. Pendekatan ini memungkinkan untuk otomatisasi proses deteksi hoaks yang lebih cepat dan efisien. |
| Metode pengolahan | Data teks diproses dengan beberapa langkah untuk meningkatkan kualitas dan konsistensi informasi yang diproses oleh model. Ini meliputi penghapusan tanda baca, normalisasi teks, dan pengubahan label menjadi format yang sesuai untuk pembelajaran mesin. |
| Arsitektur model | Model menggunakan beberapa lapisan untuk memproses dan mempelajari representasi fitur dari teks. Ini termasuk lapisan TextVectorization untuk vektorisasi teks, lapisan Embedding untuk merumuskan kata-kata menjadi vektor numerik, Conv1D untuk ekstraksi fitur spasial, dan lapisan Dense dengan Dropout untuk pengurangan overfitting. Arsitektur ini dipilih untuk memaksimalkan pemahaman dan pemodelan dari narasi berita. |
| Metrik evaluasi | Performa model dievaluasi menggunakan metrik loss (fungsi kerugian) dan accuracy (akurasi) pada data pelatihan dan validasi. Loss mengindikasikan seberapa baik model memprediksi hasil, sementara accuracy menunjukkan persentase prediksi yang benar dari total prediksi. Kedua metrik ini digunakan untuk memantau dan memperbaiki performa model selama pengembangan. |
| Performa model | Setelah melalui proses pelatihan dan validasi, model yang dihasilkan mencapai loss sebesar 0.2798 dan akurasi sebesar 0.8961 pada data pelatihan. Pada data validasi, model mencapai loss sebesar 0.6459 dan akurasi sebesar 0.8118. Hasil ini menunjukkan bahwa model dapat mengklasifikasikan berita dengan tingkat keakuratan yang tinggi, dengan potensi untuk digunakan dalam mendeteksi hoaks secara efektif. |

