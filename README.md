# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut menghadapi tingkat dropout siswa yang cukup tinggi, yang berdampak negatif terhadap reputasi institusi dan efektivitas sistem pendidikan yang dijalankan. Tanpa adanya sistem deteksi dini, sulit bagi pihak kampus untuk melakukan intervensi secara tepat waktu dan efektif. Oleh karena itu, diperlukan solusi berbasis data untuk mengidentifikasi mahasiswa yang berisiko tinggi melakukan dropout sejak dini agar dapat diberikan dukungan yang tepat sasaran.

### Permasalahan Bisnis

1. Tingginya tingkat dropout mahasiswa di Jaya Jaya Institut berdampak negatif terhadap reputasi dan performa institusi.
2. Tidak adanya sistem deteksi dini untuk mengidentifikasi mahasiswa yang berisiko tinggi melakukan dropout.
3. Keterlambatan dalam memberikan intervensi atau bimbingan menyebabkan potensi mahasiswa tidak dapat berkembang secara optimal.
4. Kurangnya pemanfaatan data historis mahasiswa untuk analisis dan pengambilan keputusan yang berbasis data.
5. Diperlukan solusi teknologi berbasis data science untuk meningkatkan efektivitas program dukungan akademik.

### Cakupan Proyek

1. Eksplorasi dan pemahaman data historis mahasiswa
2. Identifikasi fitur-fitur penting yang berkorelasi dengan risiko dropout
2. Pengembangan model machine learning untuk memprediksi risiko dropout mahasiswa.
3. Pembuatan business dashboard interaktif menggunakan Streamlit untuk membantu memantau dan menganalisis faktor-faktor dropout.
4. Rekomendasi strategi intervensi berdasarkan hasil model

### Persiapan

Sumber data: Dataset Student performance data.csv yang berisi informasi demografis mahasiswa.

Setup environment:

```
pip install -r requirements.txt
```

## Business Dashboard

Dashboard bisnis untuk monitoring dropout rate Jaya Jaya institut telah dikembangkan menggunakan Streamlit. Dashboard ini menyediakan visualisasi interaktif dan analisis komprehensif mengenai faktor-faktor yang mempengaruhi dropout.

Fitur utama dashboard:

1. **Overview** - Menampilkan metrik penting seperti distribusi dropout dan performa semester vs dropout
2. **Data Exploration** - Memungkinkan pengguna untuk melihat data mentah dan melakukan eksplorasi data mandiri
3. **Model Performance** - Menampilkan performa model secara keseluruhan
4. **Prediction** - Memungkinkan pengguna untuk melakukan prediksi dropout
5. **Batch Prediction** - Memungkinkan pengguna untuk melakukan prediksi dengan input berupa batch data
6. **About** - Menampilkan tentang insight yang di dapat pada dashboard dan cara penggunaan dashboard

Untuk menjalankan dashboard:
```
streamlit run dashboard.py
```
atau akses di streamlit community cloud:
```
https://agungknwn-students-performance-dashboard.streamlit.app/
```
```
```

## Conclusion

Dari hasil analisis data dan pembuatan dashboard, ditemukan beberapa insight penting mengenai faktor-faktor yang mempengaruhi tingkat dropout di Jaya Jaya Institut:

1. Model prediktif berhasil mengidentifikasi mahasiswa yang berisiko tinggi untuk dropout sejak semester pertama.
2. Intervensi dini menjadi kunci utama untuk mencegah mahasiswa keluar dari institusi.
3. Penerapan sistem monitoring dan dashboard peringatan dini membantu dosen pembimbing dalam mengambil tindakan cepat.
4. Strategi mentoring dan bimbingan akademik yang dipersonalisasi dapat meningkatkan retensi dan performa belajar mahasiswa.
5. Program bantuan keuangan juga penting untuk mendukung mahasiswa dengan kesulitan ekonomi.
6. Efektivitas intervensi perlu terus dipantau dan dievaluasi agar sistem bisa ditingkatkan secara berkelanjutan.
7. Pengumpulan data tambahan dan pengembangan model yang lebih kompleks akan meningkatkan akurasi prediksi di masa depan.

Dashboard yang dikembangkan memberikan visualisasi interaktif yang memudahkan departemen HR untuk memantau faktor-faktor attrition secara real-time dan membuat keputusan berbasis data untuk mengurangi tingkat attrition.

### Rekomendasi Action Items

Berdasarkan analisis data dan insight dari dashboard, berikut adalah rekomendasi action items untuk mengurangi tingkat attrition di Jaya Jaya Maju:

- Fokus pada intervensi dini untuk mahasiswa dengan performa rendah di semester pertama
- Bangun sistem dukungan tambahan untuk mahasiswa yang berisiko
- Ciptakan sesi check-in rutin dan peluang mentoring
- Kembangkan rencana akademik yang dipersonalisasi untuk mahasiswa yang menunjukkan tanda-tanda peringatan
- Pertimbangkan program bantuan keuangan bagi mahasiswa dengan kesulitan ekonomiMelakukan evaluasi dan perbaikan kebijakan lembur, termasuk kompensasi lembur dan batasan jam kerja tambahan.
