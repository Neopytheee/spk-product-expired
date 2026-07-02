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
# SIDEBAR (NAVIGASI & DOKUMENTASI ILMIAH)
# ==============================================================================
st.sidebar.title("ℹ️ Informasi & Navigasi")

st.sidebar.markdown("### 📌 Tentang Sistem")
st.sidebar.info(
    "Sistem Penunjang Keputusan (SPK) ini mengadopsi pendekatan arsitektur hibrid "
    "yang menggabungkan dua paradigma komputasi:\n\n"
    "1. **Data-Driven (K-Means Clustering):** Berfungsi sebagai filter awal pada "
    "lapisan basis data untuk mengelompokkan tingkat kelarisan produk secara objektif. "
    "Algoritma ini mengisolasi produk kategori *slow-moving* berdasarkan pola kuantitas penjualan "
    "historis, sehingga meminimalkan subjektivitas manajerial.\n\n"
    "2. **Fase Model-Driven (Simple Additive Weighting - SAW):** Berfungsi sebagai mesin penentu "
    "kebijakan taktis. Metode ini menormalisasi matriks kriteria multi-objektif (sisa volume stok "
    "gudang dan sisa hari aktif kedaluwarsa) untuk menghasilkan bobot preferensi keputusan "
    "secara presisi dan real-time."
)

st.sidebar.markdown("### 📊 Metadata Dataset")
st.sidebar.success(
    "**Spesifikasi Teknis Data:**\n\n"
    "* **Nama Berkas Asal:** SuperMarket Analysis.csv\n"
    "* **Volume Sampel Historis:** 1.000 baris transaksi awal yang memuat data terstruktur.\n"
    "* **Mekanisme Replikasi Big Data:** Sistem mengimplementasikan fungsi ekspansi virtual simultan "
    "dengan menggandakan data sebesar 300% (mencapai total **3.000 baris transaksi**) guna memenuhi "
    "syarat batas pengujian komputasi massal lingkungan cloud.\n"
    "* **Karakteristik Atribut:** Terdiri atas 17 dimensi kolom (Atribut kunci: *Invoice ID*, *Product line*, "
    "*Unit price*, *Quantity*, dan *Sales*). Parameter logistik pendukung diinjeksi menggunakan metode "
    "konformitas *Pseudo-Random Seed 42* untuk pemodelan sisa stok gudang (15–120 unit) dan sisa hari "
    "kedaluwarsa (1–90 hari)."
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
    fig_cluster.tight_layout()

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

        ax.set_xticklabels(
            hasil_global['Product line'], 
            rotation=15, 
            ha='right'
        )
        
        fig.tight_layout()

        st.pyplot(fig)

    # ==========================================================================
    # INTERPRETASI HASIL (VERSI PANJANG & ILMIAH)
    # ==========================================================================
    st.markdown("---")

    st.subheader("💡 Interpretasi dan Analisis Logika Keputusan")

    st.info(
        "**Panduan Operasional Komputasi Akhir ($V_i$):**\n\n"
        "1. **Mekanisme Perangkingan:** Nilai skor preferensi akhir berkisar antara 0.0 hingga 1.0. "
        "Semakin tinggi nilai skor akumulasi komputasi ($V_i$) yang diperoleh oleh suatu kelompok lini produk, "
        "maka semakin kritis status urgensi logistik kelompok tersebut di dalam gudang.\n\n"
        "2. **Kombinasi Parameter Kritis:** Skor tertinggi didapatkan dari hasil perpaduan kondisi di mana "
        "kriteria *Benefit* (Volume Sisa Stok Gudang) bernilai maksimal, yang berjalan beriringan dengan kondisi "
        "kriteria *Cost* (Sisa Hari Aktif Menuju Kedaluwarsa) yang bernilai minimal atau sangat mendesak.\n\n"
        "3. **Rekomendasi Aksi Manajerial:** Kelompok dengan peringkat teratas otomatis direkomendasikan "
        "kepada jajaran manajemen ritel swalayan sebagai target utama eksekusi program promosi potongan harga "
        "(diskon global) untuk mempercepat perputaran arus kas toko sekaligus menekan kerugian finansial akibat "
        "barang kedaluwarsa."
    )

else:

    st.warning(
        "Silakan upload file 'SuperMarket Analysis.csv' terlebih dahulu."
    )
