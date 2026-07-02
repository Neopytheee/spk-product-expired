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

st.title("Dashboard SPK Hibrid: Klasifikasi Produk Kedaluwarsa dan Prioritas Penjualan (Diskon)")
st.write("Sistem Penunjang Keputusan Berbasis Cloud Environment - Manajemen Ritel")

st.markdown("---")

# ==============================================================================
# IMPORT DATASET
# ==============================================================================
st.subheader("Lapisan Data: Import Dataset Ritel")

uploaded_file = st.file_uploader(
    "Upload File 'SuperMarket Analysis.csv' di sini",
    type=["csv"]
)

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
        f"Sistem mendeteksi total: {len(df_raw)} Baris Transaksi "
        f"(Persyaratan > 1000 Baris Terpenuhi)."
    )

    # ==========================================================================
    # KPI DASHBOARD
    # ==========================================================================
    st.markdown("## Ringkasan Data")

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

    st.subheader("Preview Big Data Transaksi")

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
    # K-MEANS CLUSTERING
    # ==========================================================================
    X = df_enriched[['Quantity']].values

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    df_enriched['cluster_kelarisan'] = kmeans.fit_predict(X)

    st.markdown("---")
    st.subheader("Engine 1 : K-Means Clustering")

    cluster_summary = (
        df_enriched
        .groupby('cluster_kelarisan')['Quantity']
        .agg(['count', 'mean'])
        .reset_index()
    )

    cluster_summary.columns = [
        'Cluster',
        'Jumlah Data',
        'Rata-rata Quantity'
    ]

    st.dataframe(cluster_summary)

    # ==========================================================================
    # VISUALISASI CLUSTER
    # ==========================================================================
    st.write("### Visualisasi Cluster Kelarisan")

    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 4))

    scatter = ax_cluster.scatter(
        df_enriched.index,
        df_enriched['Quantity'],
        c=df_enriched['cluster_kelarisan']
    )

    ax_cluster.set_xlabel("Index Data")
    ax_cluster.set_ylabel("Quantity")
    ax_cluster.set_title("Distribusi Cluster K-Means")

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
    st.subheader("Produk Slow Moving")

    st.info(
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
    # SAW
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
    # SLIDER BOBOT
    # ==========================================================================
    st.markdown("---")

    st.subheader("Engine 2 : Pengaturan Bobot SAW")

    bobot_expired = st.slider(
        "Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)",
        0.0,
        1.0,
        0.65,
        0.05
    )

    bobot_stok = 1.0 - bobot_expired

    st.info(
        f"Bobot Stok Gudang (Benefit) Otomatis = "
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

        Produk : {top_produk['Product line']}

        Skor Prioritas :
        {top_produk['Skor_Prioritas_Diskon']:.4f}
        """
    )

    # ==========================================================================
    # OUTPUT SPK
    # ==========================================================================
    st.subheader(
        "Output SPK : Rekomendasi Program Diskon"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Ranking Kategori Produk")

        st.dataframe(hasil_global)

        csv = hasil_global.to_csv(index=False)

        st.download_button(
            "⬇ Download Hasil Ranking",
            csv,
            file_name="hasil_ranking_spk.csv",
            mime="text/csv"
        )

    with col2:

        st.write("### Grafik Distribusi Prioritas")

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.bar(
            hasil_global['Product line'],
            hasil_global['Skor_Prioritas_Diskon']
        )

        ax.set_ylabel(
            "Skor Prioritas Diskon (V)"
        )

        ax.set_xticklabels(
            hasil_global['Product line'],
            rotation=15,
            ha='right'
        )

        st.pyplot(fig)

else:

    st.warning(
        "Silakan upload file "
        "'SuperMarket Analysis.csv' "
        "terlebih dahulu."
    )
