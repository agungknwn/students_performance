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
conda create --name main python=3.9
conda activate main
pip install -r requirements.txt
```

## Business Dashboard

Dashboard bisnis untuk monitoring dropout rate Jaya Jaya institut telah dikembangkan menggunakan Streamlit. Dashboard ini menyediakan visualisasi interaktif dan analisis komprehensif mengenai faktor-faktor yang mempengaruhi dropout.

Fitur utama dashboard:

1. **Overview** - Menampilkan metrik penting seperti distribusi dropout dan overview singkat tentang data yang digunakan
2. **Prediction** - Memungkinkan pengguna untuk melakukan prediksi dropout
3. **Insight** - Menampilkan tentang insight yang di dapat pada dashboard dan cara penggunaan dashboard

Untuk menjalankan dashboard:
```
streamlit run dashboard.py
```
atau akses di streamlit community cloud:
```
https://agungknwn-jayainstitut.streamlit.app/
```

## Conclusion

Dari hasil analisis data dan pembuatan dashboard, ditemukan beberapa insight penting mengenai faktor-faktor yang mempengaruhi tingkat dropout di Jaya Jaya Institut:
- Model machine learning yang dikembangkan dapat memprediksi mahasiswa yang berisiko dropout dengan performa yang baik.
- Faktor-faktor yang paling berpengaruh dalam memprediksi dropout adalah:
    1. Performa akademik pada semester pertama dan kedua
    2. Biaya akademik
    3. Nilai masuk

Dashboard yang dikembangkan memberikan visualisasi interaktif yang memudahkan pengajar dan stakeholder terkait untuk memantau faktor-faktor dropout secara real-time dan membuat keputusan berbasis data untuk mengurangi tingkat dropout mahasiswa Jaya Jaya institut.

### Rekomendasi Action Items

Berdasarkan analisis data dan insight dari dashboard, berikut adalah rekomendasi action items untuk mengurangi tingkat droput di Jaya Jaya Institut:
1. **Sistem Deteksi Dini**: Implementasikan model ini sebagai sistem peringatan dini untuk mengidentifikasi mahasiswa berisiko dropout sejak awal.
2. **Program Bimbingan Khusus**: Berikan bimbingan akademik khusus untuk mahasiswa yang diidentifikasi berisiko tinggi.
3. **Peningkatan Dukungan**: Tingkatkan dukungan akademik, keuangan, dan sosial untuk mahasiswa berisiko.
4. **Monitoring Berkelanjutan**: Pantau secara berkelanjutan performa mahasiswa, terutama pada faktor-faktor penting penentu dropout.
5. **Evaluasi Kurikulum**: Evaluasi dan sesuaikan kurikulum untuk mata kuliah dengan tingkat kegagalan tinggi.
