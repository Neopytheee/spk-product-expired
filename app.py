import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==============================================================================
# IMPORT DATASET
# ==============================================================================
uploaded_file = st.file_uploader(
    "Upload File 'SuperMarket Analysis.csv' di sini",
    type=["csv"]
)

if uploaded_file is not None:

    df_base = pd.read_csv(uploaded_file)

    st.success(
        f"✅ File berhasil diupload. Jumlah data asli: {len(df_base)} baris"
    )

    st.write("### Preview Dataset")
    st.dataframe(df_base.head())

    # Simpan status tombol
    if "analisis_dijalankan" not in st.session_state:
        st.session_state.analisis_dijalankan = False

    if st.button("🚀 Jalankan Analisis SPK", type="primary"):
        st.session_state.analisis_dijalankan = True

    # =====================================================
    # PROSES HANYA JIKA TOMBOL DIKLIK
    # =====================================================
    if st.session_state.analisis_dijalankan:

        # =====================================================
        # EKSPANSI DATA
        # =====================================================
        df_raw = pd.concat(
            [df_base, df_base, df_base],
            ignore_index=True
        )

        df_raw["Invoice ID"] = [
            f"INV-{100000+i}"
            for i in range(len(df_raw))
        ]

        st.success(
            f"✅ Big Data Berhasil Diproses! Total {len(df_raw)} baris transaksi."
        )

        # =====================================================
        # PREPROCESSING
        # =====================================================
        np.random.seed(42)

        df_enriched = df_raw.copy()

        df_enriched["stok_tersisa"] = np.random.randint(
            15,
            120,
            size=len(df_enriched)
        )

        df_enriched["sisa_hari_expired"] = np.random.randint(
            1,
            90,
            size=len(df_enriched)
        )

        # =====================================================
        # K-MEANS
        # =====================================================
        X = df_enriched[["Quantity"]].values

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10   # lebih aman daripada 'auto'
        )

        df_enriched["cluster_kelarisan"] = (
            kmeans.fit_predict(X)
        )

        # =====================================================
        # AMBIL CLUSTER SLOW MOVING
        # =====================================================
        slow_moving_cluster = (
            df_enriched
            .groupby("cluster_kelarisan")["Quantity"]
            .mean()
            .idxmin()
        )

        df_slow_moving = (
            df_enriched[
                df_enriched["cluster_kelarisan"]
                == slow_moving_cluster
            ]
            .copy()
        )

        # =====================================================
        # SLIDER BOBOT
        # =====================================================
        st.markdown("---")
        st.subheader("Aturan Manajemen: Pengaturan Bobot SPK")

        bobot_expired = st.slider(
            "Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05
        )

        bobot_stok = 1.0 - bobot_expired

        st.info(
            f"Bobot Volume Stok Gudang (Benefit): {bobot_stok:.2f}"
        )

        # =====================================================
        # NORMALISASI SAW
        # =====================================================
        max_stok = df_slow_moving["stok_tersisa"].max()
        min_expired = df_slow_moving["sisa_hari_expired"].min()

        df_slow_moving["R_stok"] = (
            df_slow_moving["stok_tersisa"] / max_stok
        )

        df_slow_moving["R_expired"] = (
            min_expired /
            df_slow_moving["sisa_hari_expired"]
        )

        # =====================================================
        # HITUNG SKOR SAW
        # =====================================================
        df_slow_moving["Skor_Prioritas_Diskon"] = (
            (df_slow_moving["R_stok"] * bobot_stok)
            +
            (df_slow_moving["R_expired"] * bobot_expired)
        )

        hasil_global = (
            df_slow_moving
            .groupby("Product line")
            [
                [
                    "stok_tersisa",
                    "sisa_hari_expired",
                    "Skor_Prioritas_Diskon"
                ]
            ]
            .mean()
            .sort_values(
                by="Skor_Prioritas_Diskon",
                ascending=False
            )
            .reset_index()
        )

        # =====================================================
        # OUTPUT
        # =====================================================
        st.markdown("---")
        st.subheader(
            "Output SPK: Rekomendasi Utama Program Diskon"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Tabel Ranking Produk")
            st.dataframe(hasil_global)

        with col2:
            st.write("### Grafik Distribusi Prioritas")

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                hasil_global["Product line"],
                hasil_global["Skor_Prioritas_Diskon"]
            )

            ax.set_ylabel(
                "Skor Prioritas Diskon"
            )

            plt.xticks(rotation=15)

            st.pyplot(fig)

else:
    st.warning(
        "Silakan upload file 'SuperMarket Analysis.csv' terlebih dahulu."
    )
