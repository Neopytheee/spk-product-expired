uploaded_file = st.file_uploader(
    "Upload File 'SuperMarket Analysis.csv' di sini",
    type=["csv"]
)

if uploaded_file is not None:

    df_base = pd.read_csv(uploaded_file)

    st.success(f"File berhasil diupload. Jumlah data asli: {len(df_base)} baris")

    st.write("### Preview Dataset")
    st.dataframe(df_base.head())

    # Tombol untuk menjalankan proses analisis
    if st.button("🚀 Jalankan Analisis SPK", type="primary"):

        # =====================================================
        # EKSPANSI DATA
        # =====================================================
        df_raw = pd.concat(
            [df_base, df_base, df_base],
            ignore_index=True
        )

        df_raw['Invoice ID'] = [
            f"INV-{100000 + i}"
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
        df_enriched['stok_tersisa'] = np.random.randint(
            15, 120, size=len(df_enriched)
        )
        df_enriched['sisa_hari_expired'] = np.random.randint(
            1, 90, size=len(df_enriched)
        )

        # =====================================================
        # K-MEANS
        # =====================================================
        X = df_enriched[['Quantity']].values

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init='auto'
        )

        df_enriched['cluster_kelarisan'] = kmeans.fit_predict(X)

        # =====================================================
        # SAW
        # =====================================================
        slow_moving_cluster = (
            df_enriched
            .groupby('cluster_kelarisan')['Quantity']
            .mean()
            .idxmin()
        )

        df_slow_moving = df_enriched[
            df_enriched['cluster_kelarisan']
            == slow_moving_cluster
        ].copy()

        max_stok = df_slow_moving['stok_tersisa'].max()
        min_expired = df_slow_moving['sisa_hari_expired'].min()

        df_slow_moving['R_stok'] = (
            df_slow_moving['stok_tersisa'] / max_stok
        )

        df_slow_moving['R_expired'] = (
            min_expired /
            df_slow_moving['sisa_hari_expired']
        )

        # =====================================================
        # BOBOT
        # =====================================================
        bobot_expired = st.slider(
            "Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)",
            0.0,
            1.0,
            0.65,
            0.05
        )

        bobot_stok = 1 - bobot_expired

        df_slow_moving['Skor_Prioritas_Diskon'] = (
            (df_slow_moving['R_stok'] * bobot_stok) +
            (df_slow_moving['R_expired'] * bobot_expired)
        )

        hasil_global = (
            df_slow_moving
            .groupby('Product line')
            [['stok_tersisa',
              'sisa_hari_expired',
              'Skor_Prioritas_Diskon']]
            .mean()
            .sort_values(
                by='Skor_Prioritas_Diskon',
                ascending=False
            )
            .reset_index()
        )

        # =====================================================
        # OUTPUT
        # =====================================================
        st.subheader(
            "Output SPK: Rekomendasi Utama Program Diskon"
        )

        st.dataframe(hasil_global)

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            hasil_global['Product line'],
            hasil_global['Skor_Prioritas_Diskon']
        )

        plt.xticks(rotation=15)

        st.pyplot(fig)

else:
    st.warning(
        "Silakan upload file CSV terlebih dahulu."
    )
