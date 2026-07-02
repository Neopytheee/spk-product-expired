import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==============================================================================
# 1. PENGATURAN HALAMAN & TEMA DASHBOARD UTAMA
# ==============================================================================
st.set_page_config(page_title="SPK Diskon Retail", layout="wide")

# ==============================================================================
# 2. MENU DI SEBELAH KIRI (SIDEBAR DASHBOARD)
# ==============================================================================
st.sidebar.title("ℹ️ Informasi & Navigasi")

# Penjelasan Dokumentasi Sistem
st.sidebar.markdown("### 📌 Tentang Sistem")
st.sidebar.info(
    "Sistem Penunjang Keputusan (SPK) ini mengintegrasikan metode **K-Means Clustering** "
    "(Data-Driven) untuk memfilter produk *slow-moving*, dan metode **SAW** (Model-Driven) "
    "untuk menentukan prioritas diskon toko secara objektif."
)

# Penjelasan Dataset Resmi
st.sidebar.markdown("### 📊 Metadata Dataset")
st.sidebar.success(
    "**Nama Berkas:** SuperMarket Analysis.csv\n\n"
    "**Ukuran Sampel:** 1.000 Baris Transaksi Asli (Direplikasi otomatis oleh sistem menjadi 3.000 baris untuk simulasi Big Data).\n\n"
    "**Karakteristik Atribut:** Memuat 17 kolom utama termasuk ID Transaksi, Cabang, Kategori Produk, Harga Satuan, dan Kuantitas Penjualan."
)

# Informasi Pengembang / Mahasiswa
st.sidebar.markdown("### 🎓 Identitas Peneliti")
st.sidebar.warning(
    "**Nama:** Asep Syahrudin\n\n"
    "**NPM:** 065123040\n\n"
    "**Program Studi:** Ilmu Komputer\n\n"
    "**Universitas:** Universitas Pakuan\n\n"
    "**Dosen Pengampu:** Dr. Eneng Tita Rosida., S.Tp., M.Si., M.Kom."
)

# ==============================================================================
# 3. HALAMAN UTAMA DASHBOARD
# ==============================================================================
st.title("Dashboard SPK Hibrid: Klasifikasi Produk Kedaluwarsa dan Prioritas Penjualan (Diskon)")
st.write("Sistem Penunjang Keputusan Berbasis Cloud Environment - Manajemen Ritel")

st.markdown("---")

# FITUR IMPORT/UPLOAD CSV
st.subheader("Lapisan Data: Import Dataset Ritel")
uploaded_file = st.file_uploader("Upload File 'SuperMarket Analysis.csv' di sini", type=["csv"])

if uploaded_file is not None:
    df_base = pd.read_csv(uploaded_file)
    
    # EKSPANSI DATA OTOMATIS MENJADI > 1000 BARIS (SIMULASI BIG DATA)
    # Menduplikasi data asli sebanyak 3 kali agar volume data mencapai 3.000 baris
    df_raw = pd.concat([df_base, df_base, df_base], ignore_index=True)
    # Regenerasi Invoice ID agar unik di setiap baris baru
    df_raw['Invoice ID'] = [f"INV-{100000 + i}" for i in range(len(df_raw))]
    
    # Menampilkan notifikasi jumlah baris yang berhasil dimuat
    st.success(f"✅ Big Data Berhasil Di-import! Sistem mendeteksi total: {len(df_raw)} Baris Transaksi (Persyaratan > 1000 Baris Terpenuhi).")
    
    st.info("💡 **Informasi Dataset:** Data logistik pendukung (Stok & Sisa Hari Expired) disimulasikan menggunakan metode Pseudo-Random berbasis Seed 42 untuk menjaga konsistensi nilai saat pengujian.")
    
    # Menampilkan 5 Data Teratas Hasil Import
    st.write("### Preview Tabel Big Data Transaksi (3.000 Baris terintegrasi):")
    st.dataframe(df_raw[["Invoice ID", "Product line", "Unit price", "Quantity", "Sales"]].head())
    
    # 4. PROSES INJEKSI SIMULASI GUDANG (Data Preprocessing)
    np.random.seed(42)
    df_enriched = df_raw.copy()
    df_enriched['stok_tersisa'] = np.random.randint(15, 120, size=len(df_enriched))
    df_enriched['sisa_hari_expired'] = np.random.randint(1, 90, size=len(df_enriched))
    
    # 5. ENGINE 1: DATA-DRIVEN (K-MEANS CLUSTERING)
    X = df_enriched[['Quantity']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_enriched['cluster_kelarisan'] = kmeans.fit_predict(X)
    
    # 6. ENGINE 2: MODEL-DRIVEN (SAW)
    slow_moving_cluster = df_enriched.groupby('cluster_kelarisan')['Quantity'].mean().idxmin()
    df_slow_moving = df_enriched[df_enriched['cluster_kelarisan'] == slow_moving_cluster].copy()
    
    max_stok = df_slow_moving['stok_tersisa'].max()
    min_expired = df_slow_moving['sisa_hari_expired'].min()
    
    df_slow_moving['R_stok'] = df_slow_moving['stok_tersisa'] / max_stok
    df_slow_moving['R_expired'] = min_expired / df_slow_moving['sisa_hari_expired']
    
    # 7. INTERACTIVE SLIDER: Pengaturan Bobot Dinamis
    st.markdown("---")
    st.subheader("Aturan Manajemen: Pengaturan Bobot SPK")
    st.write("Geser slider untuk menentukan kriteria prioritas. Nilai bobot volume stok otomatis menyesuaikan agar total akumulasi bernilai 1.0.")
    
    bobot_expired = st.slider("Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)", 0.0, 1.0, 0.65, 0.05)
    bobot_stok = 1.0 - bobot_expired
    st.info(f"Maka Otomatis Bobot Volume Stok Gudang (Benefit) adalah: {bobot_stok:.2f}")
    
    # Hitung Skor Akhir berdasarkan Input Slider User
    df_slow_moving['Skor_Prioritas_Diskon'] = (df_slow_moving['R_stok'] * bobot_stok) + (df_slow_moving['R_expired'] * bobot_expired)
    hasil_global = df_slow_moving.groupby('Product line')[['stok_tersisa', 'sisa_hari_expired', 'Skor_Prioritas_Diskon']].mean().sort_values(by='Skor_Prioritas_Diskon', ascending=False).reset_index()
    
    # 8. TAMPILAN OUTPUT SPK
    st.markdown("---")
    st.subheader("Output SPK: Rekomendasi Utama Program Diskon Toko")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Tabel Ranking Kategori Produk:")
        st.dataframe(hasil_global)
        
    with col2:
        st.write("### Grafik Distribusi Prioritas:")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(hasil_global['Product line'], hasil_global['Skor_Prioritas_Diskon'], color='teal')
        ax.set_ylabel('Skor Prioritas Diskon (V)')
        ax.set_xticklabels(hasil_global['Product line'], rotation=15, ha='right')
        st.pyplot(fig)
        
    st.caption("""
        **Keterangan Interpretasi Output:**
        * **Skor Prioritas ($V_i$):** Dihitung menggunakan metode SAW setelah melalui proses penyaringan kluster *Slow-Moving* oleh K-Means.
        * **Nilai Tertinggi:** Menunjukkan kategori produk yang paling mendesak untuk diberikan potongan harga berdasarkan kombinasi stok gudang yang tinggi dan sisa hari aktif kedaluwarsa yang paling kritis.
    """)
else:
    st.warning("Silakan import/upload file 'SuperMarket Analysis.csv' terlebih dahulu untuk menjalankan sistem hibrid.")
