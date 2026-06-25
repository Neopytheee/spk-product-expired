%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Judul & Pengaturan Aplikasi di Web Dashboard
st.set_page_config(page_title="SPK Diskon Retail", layout="wide")
st.title("📊 Dashboard SPK Hibrid: Prioritas Diskon Produk Kedaluwarsa (K-Means + SAW)")
st.write("Sistem Pendukung Keputusan Berbasis Cloud Environment - Manajemen Ritel")

st.markdown("---")

# 2. FITUR IMPORT/UPLOAD CSV (Sama seperti fungsi tombol Import di Kelompok Teman Anda)
st.subheader("📥 Lapisan Data: Import Dataset Ritel")
uploaded_file = st.file_uploader("Upload File 'SuperMarket Analysis.csv' di sini", type=["csv"])

if uploaded_file is not None:
    # Mengambil data dari file yang di-import oleh user
    df_raw = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Berhasil Di-import ke Sistem!")
    
    # Menampilkan 5 Data Teratas Hasil Import
    st.write("### Preview Data Awal:")
    st.dataframe(df_raw[["Invoice ID", "Product line", "Unit price", "Quantity", "Sales"]].head())
    
    # 3. PROSES INJEKSI SIMULASI GUDANG (Data Preprocessing)
    np.random.seed(42)
    df_enriched = df_raw.copy()
    df_enriched['stok_tersisa'] = np.random.randint(15, 120, size=len(df_enriched))
    df_enriched['sisa_hari_expired'] = np.random.randint(1, 90, size=len(df_enriched))
    
    # 4. ENGINE 1: DATA-DRIVEN (K-MEANS CLUSTERING)
    X = df_enriched[['Quantity']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_enriched['cluster_kelarisan'] = kmeans.fit_predict(X)
    
    # 5. ENGINE 2: MODEL-DRIVEN (SAW)
    slow_moving_cluster = df_enriched.groupby('cluster_kelarisan')['Quantity'].mean().idxmin()
    df_slow_moving = df_enriched[df_enriched['cluster_kelarisan'] == slow_moving_cluster].copy()
    
    max_stok = df_slow_moving['stok_tersisa'].max()
    min_expired = df_slow_moving['sisa_hari_expired'].min()
    
    df_slow_moving['R_stok'] = df_slow_moving['stok_tersisa'] / max_stok
    df_slow_moving['R_expired'] = min_expired / df_slow_moving['sisa_hari_expired']
    
    # 6. INTERACTIVE SLIDER: Fitur Premium agar User bisa Atur Bobot Sendiri di UI!
    st.markdown("---")
    st.subheader("⚙️ Aturan Manajemen: Pengaturan Bobot SPK")
    bobot_expired = st.slider("Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)", 0.0, 1.0, 0.65, 0.05)
    bobot_stok = 1.0 - bobot_expired
    st.info(f"Maka Otomatis Bobot Volume Stok Gudang (Benefit) adalah: {bobot_stok:.2f}")
    
    # Hitung Skor Akhir berdasarkan Input Slider User
    df_slow_moving['Skor_Prioritas_Diskon'] = (df_slow_moving['R_stok'] * bobot_stok) + (df_slow_moving['R_expired'] * bobot_expired)
    
    # Agregasi Hasil Rekomendasi Global per Lini Produk
    hasil_global = df_slow_moving.groupby('Product line')[['stok_tersisa', 'sisa_hari_expired', 'Skor_Prioritas_Diskon']].mean().sort_values(by='Skor_Prioritas_Diskon', ascending=False).reset_index()
    
    # 7. TAMPILAN OUTPUT SPK (Visualisasi Real-time)
    st.markdown("---")
    st.subheader("🎯 Output SPK: Rekomendasi Utama Program Diskon Toko")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Tabel Ranking Kategori Produk:")
        st.dataframe(hasil_global)
        
    with col2:
        st.write("### Grafik Distribusi Prioritas:")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(hasil_global['Product line'], hasil_global['Skor_Prioritas_Diskon'], color='teal')
        ax.set_ylabel('Skor Prioritas Diskon (V)')
        plt.xticks(rotation=15)
        st.pyplot(fig)

else:
    st.warning("⚠️ Silakan import/upload file 'SuperMarket Analysis.csv' terlebih dahulu untuk menjalankan sistem hibrid.")
