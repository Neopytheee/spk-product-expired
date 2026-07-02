import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="SPK Diskon Retail",
    layout="wide"
)

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("ℹ️ Informasi & Navigasi")

st.sidebar.markdown("### 📌 Tentang Sistem")
st.sidebar.info(
    "Sistem Penunjang Keputusan (SPK) ini mengadopsi pendekatan arsitektur hibrid "
    "yang menggabungkan dua paradigma komputasi:\n\n"
    "1. **Data-Driven (K-Means Clustering)** untuk mengidentifikasi produk slow-moving.\n\n"
    "2. **Model-Driven (Simple Additive Weighting / SAW)** untuk menentukan prioritas diskon "
    "berdasarkan stok dan masa kedaluwarsa."
)

st.sidebar.markdown("### 📊 Metadata Dataset")
st.sidebar.success(
    "**Dataset:** SuperMarket Analysis.csv\n\n"
    "- Dataset awal ±1000 transaksi\n"
    "- Diekspansi menjadi ±3000 transaksi\n"
    "- Simulasi stok gudang dan expired menggunakan Seed 42\n"
    "- Digunakan untuk pengujian SPK berbasis cloud"
)

st.sidebar.markdown("### 🎓 Identitas Peneliti")
st.sidebar.warning(
    "**Nama:** Asep Syahrudin\n\n"
    "**NPM:** 065123040\n\n"
    "**Program Studi:** Ilmu Komputer\n\n"
    "**Universitas:** Universitas Pakuan\n\n"
    "**Dosen:** Dr. Eneng Tita Rosida, S.Tp., M.Si., M.Kom."
)

# ==============================================================================
# HEADER
# ==============================================================================
st.title(
    "Dashboard SPK Hibrid: Klasifikasi Produk Kedaluwarsa dan Prioritas Penjualan (Diskon)"
)

st.write(
    "Sistem Penunjang Keputusan Berbasis Cloud Environment - Manajemen Ritel"
)

st.markdown("---")

# ==============================================================================
# UPLOAD DATA
# ==============================================================================
st.subheader("📂 Lapisan Data: Import Dataset Ritel")

uploaded_file = st.file_uploader(
    "Upload File SuperMarket Analysis.csv",
    type=["csv"]
)

# ==============================================================================
# PROSES UTAMA
# ==============================================================================
if uploaded_file is not None:

    df_base = pd.read_csv(uploaded_file)

    # ==========================================================================
    # BIG DATA EXPANSION
    # ==========================================================================
    df_raw = pd.concat(
        [df_base, df_base, df_base],
        ignore_index=True
    )

    df_raw['Invoice ID'] = [
        f"INV-{100000+i}"
        for i in range(len(df_raw))
    ]

    st.success(
        f"✅ Big Data Berhasil Di-import! "
        f"Sistem mendeteksi total {len(df_raw)} transaksi."
    )

    st.info(
        "💡 Data stok gudang dan sisa hari expired "
        "disimulasikan menggunakan Pseudo-Random Seed 42."
    )

    # ==========================================================================
    # KPI DASHBOARD
    # ==========================================================================
    st.markdown("## 📊 Ringkasan Statistik Big Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transaksi",
            len(df_raw)
        )

    with col2:
        st.metric(
            "Total Penjualan",
            f"${df_raw['Sales'].sum():,.0f}"
        )

    with col3:
        st.metric(
            "Kategori Produk",
            df_raw['Product line'].nunique()
        )

    with col4:
        st.metric(
            "Rata-rata Quantity",
            round(df_raw['Quantity'].mean(), 2)
        )

    # ==========================================================================
    # PREVIEW DATA
    # ==========================================================================
    st.markdown("---")

    st.subheader("📄 Preview Dataset")

    st.dataframe(
        df_raw[
            [
                "Invoice ID",
                "Product line",
                "Unit price",
                "Quantity",
                "Sales"
            ]
        ].head()
    )

    # ==========================================================================
    # DATA ENRICHMENT
    # ==========================================================================
    np.random.seed(42)

    df_enriched = df_raw.copy()

    df_enriched['stok_tersisa'] = np.random.randint(
        15,
        120,
        size=len(df_enriched)
    )

    df_enriched['sisa_hari_expired'] = np.random.randint(
        1,
        90,
        size=len(df_enriched)
    )

    # ==========================================================================
    # ENGINE 1 : K-MEANS
    # ==========================================================================
    X = df_enriched[['Quantity']].values

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    df_enriched['cluster_kelarisan'] = kmeans.fit_predict(X)

    st.markdown("---")
    st.subheader("📌 Engine 1 : K-Means Clustering")

    cluster_summary = (
        df_enriched
        .groupby('cluster_kelarisan')['Quantity']
        .agg(['count', 'mean'])
        .reset_index()
    )

    cluster_summary.columns = [
        "Cluster",
        "Jumlah Data",
        "Rata-rata Quantity"
    ]

    st.dataframe(cluster_summary)

    # ==========================================================================
    # VISUALISASI CLUSTER
    # ==========================================================================
    st.write("### Visualisasi Cluster Kelarisan")

    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 4))

    ax_cluster.scatter(
        df_enriched.index,
        df_enriched['Quantity'],
        c=df_enriched['cluster_kelarisan']
    )

    ax_cluster.set_title(
        "Distribusi Hasil K-Means Clustering"
    )

    ax_cluster.set_xlabel("Index Data")
    ax_cluster.set_ylabel("Quantity")

    st.pyplot(fig_cluster)

    # ==========================================================================
    # IDENTIFIKASI SLOW MOVING
    # ==========================================================================
    slow_moving_cluster = (
        df_enriched
        .groupby('cluster_kelarisan')['Quantity']
        .mean()
        .idxmin()
    )

    df_slow_moving = (
        df_enriched[
            df_enriched['cluster_kelarisan']
            == slow_moving_cluster
        ]
        .copy()
    )

    st.markdown("---")

    st.subheader("🚚 Identifikasi Slow Moving Product")

    st.success(
        f"Cluster Slow Moving yang terdeteksi: "
        f"Cluster {slow_moving_cluster}"
    )

    st.dataframe(
        df_slow_moving[
            [
                'Product line',
                'Quantity',
                'stok_tersisa',
                'sisa_hari_expired'
            ]
        ].head(20)
    )

    # ==========================================================================
    # NORMALISASI SAW
    # ==========================================================================
    max_stok = df_slow_moving['stok_tersisa'].max()
    min_expired = df_slow_moving['sisa_hari_expired'].min()

    df_slow_moving['R_stok'] = (
        df_slow_moving['stok_tersisa']
        / max_stok
    )

    df_slow_moving['R_expired'] = (
        min_expired
        / df_slow_moving['sisa_hari_expired']
    )

    # ==========================================================================
    # BOBOT DINAMIS
    # ==========================================================================
    st.markdown("---")

    st.subheader("⚙️ Engine 2 : Pengaturan Bobot SAW")

    bobot_expired = st.slider(
        "Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)",
        0.0,
        1.0,
        0.65,
        0.05
    )

    bobot_stok = 1.0 - bobot_expired

    st.info(
        f"Bobot Volume Stok Gudang (Benefit): "
        f"{bobot_stok:.2f}"
    )

    # ==========================================================================
    # PERHITUNGAN SAW
    # ==========================================================================
    df_slow_moving['Skor_Prioritas_Diskon'] = (
        (df_slow_moving['R_stok'] * bobot_stok)
        +
        (df_slow_moving['R_expired'] * bobot_expired)
    )

    hasil_global = (
        df_slow_moving
        .groupby('Product line')
        [
            [
                'stok_tersisa',
                'sisa_hari_expired',
                'Skor_Prioritas_Diskon'
            ]
        ]
        .mean()
        .sort_values(
            by='Skor_Prioritas_Diskon',
            ascending=False
        )
        .reset_index()
    )

    # ==========================================================================
    # PRIORITAS UTAMA
    # ==========================================================================
    st.markdown("---")

    top_produk = hasil_global.iloc[0]

    st.success(
        f"""
        🎯 PRIORITAS UTAMA PROGRAM DISKON

        Product Line : {top_produk['Product line']}

        Skor Prioritas :
        {top_produk['Skor_Prioritas_Diskon']:.4f}
        """
    )

    # ==========================================================================
    # OUTPUT SPK
    # ==========================================================================
    st.markdown("---")

    st.subheader(
        "📈 Output SPK: Rekomendasi Program Diskon"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Ranking Kategori Produk")

        st.dataframe(hasil_global)

        csv = hasil_global.to_csv(index=False)

        st.download_button(
            label="⬇ Download Hasil Ranking",
            data=csv,
            file_name="hasil_ranking_spk.csv",
            mime="text/csv"
        )

    with col2:

        st.write("### Grafik Distribusi Prioritas")

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.bar(
            hasil_global['Product line'],
            hasil_global['Skor_Prioritas_Diskon'],
            color='teal'
        )

        ax.set_ylabel(
            "Skor Prioritas Diskon (V)"
        )

        plt.xticks(
            rotation=15,
            ha='right'
        )

        st.pyplot(fig)

    # ==========================================================================
    # INTERPRETASI HASIL
    # ==========================================================================
    st.markdown("---")

    st.subheader("💡 Interpretasi dan Analisis Keputusan")

    st.info(
        "Semakin tinggi skor prioritas diskon (Vi), semakin tinggi urgensi "
        "produk untuk masuk program diskon. Nilai tinggi diperoleh dari "
        "kombinasi stok gudang yang besar dan masa kedaluwarsa yang semakin dekat. "
        "Produk dengan skor tertinggi direkomendasikan sebagai prioritas utama "
        "program promosi guna mengurangi risiko kerugian akibat barang kedaluwarsa."
    )

else:

    st.warning(
        "Silakan upload file 'SuperMarket Analysis.csv' terlebih dahulu."
    )
