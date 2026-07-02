# ==========================================================================
# TABS DASHBOARD
# ==========================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data",
    "🤖 Engine 1 - K-Means",
    "🚚 Slow Moving",
    "⚙️ Engine 2 - SAW",
    "📈 Hasil SPK"
])

# ==============================================================================
# TAB 1 : DATA
# ==============================================================================
with tab1:

    st.subheader("📊 Ringkasan Statistik Big Data")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transaksi",
        len(df_raw)
    )

    col2.metric(
        "Total Penjualan",
        f"${df_raw['Sales'].sum():,.0f}"
    )

    col3.metric(
        "Kategori Produk",
        df_raw['Product line'].nunique()
    )

    col4.metric(
        "Rata-rata Quantity",
        round(df_raw['Quantity'].mean(), 2)
    )

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
        ].head(20)
    )

# ==============================================================================
# TAB 2 : ENGINE 1 K-MEANS
# ==============================================================================
with tab2:

    st.subheader("🤖 Engine 1 : K-Means Clustering")

    st.dataframe(cluster_summary)

    st.markdown("---")

    st.write("### Visualisasi Cluster")

    fig_cluster, ax_cluster = plt.subplots(figsize=(8,4))

    ax_cluster.scatter(
        df_enriched.index,
        df_enriched['Quantity'],
        c=df_enriched['cluster_kelarisan']
    )

    ax_cluster.set_title(
        "Distribusi Hasil K-Means"
    )

    ax_cluster.set_xlabel("Index Data")
    ax_cluster.set_ylabel("Quantity")

    st.pyplot(fig_cluster)

# ==============================================================================
# TAB 3 : SLOW MOVING
# ==============================================================================
with tab3:

    st.subheader(
        "🚚 Identifikasi Slow Moving Product"
    )

    st.success(
        f"Cluster Slow Moving : {slow_moving_cluster}"
    )

    st.dataframe(
        df_slow_moving[
            [
                'Product line',
                'Quantity',
                'stok_tersisa',
                'sisa_hari_expired'
            ]
        ].head(50)
    )

# ==============================================================================
# TAB 4 : ENGINE 2 SAW
# ==============================================================================
with tab4:

    st.subheader(
        "⚙️ Engine 2 : Simple Additive Weighting (SAW)"
    )

    st.write(
        "Atur bobot prioritas kedaluwarsa menggunakan slider berikut."
    )

    bobot_expired = st.slider(
        "Bobot Kriteria Sisa Hari Kedaluwarsa (Cost)",
        0.0,
        1.0,
        0.65,
        0.05,
        key="slider_saw"
    )

    bobot_stok = 1.0 - bobot_expired

    st.info(
        f"Bobot Volume Stok Gudang (Benefit): {bobot_stok:.2f}"
    )

    df_slow_moving['Skor_Prioritas_Diskon'] = (
        (df_slow_moving['R_stok'] * bobot_stok)
        +
        (df_slow_moving['R_expired'] * bobot_expired)
    )

    st.markdown("---")

    st.write("### Preview Perhitungan SAW")

    st.dataframe(
        df_slow_moving[
            [
                'Product line',
                'R_stok',
                'R_expired',
                'Skor_Prioritas_Diskon'
            ]
        ].head(30)
    )

# ==============================================================================
# TAB 5 : HASIL SPK
# ==============================================================================
with tab5:

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

    top_produk = hasil_global.iloc[0]

    st.success(
        f"""
🎯 PRIORITAS UTAMA PROGRAM DISKON

Product Line : {top_produk['Product line']}

Skor Prioritas : {top_produk['Skor_Prioritas_Diskon']:.4f}
"""
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Ranking Kategori Produk")

        st.dataframe(hasil_global)

        csv = hasil_global.to_csv(index=False)

        st.download_button(
            "⬇ Download Hasil Ranking",
            csv,
            "hasil_ranking_spk.csv",
            "text/csv"
        )

    with col2:

        st.write("### Grafik Distribusi Prioritas")

        fig, ax = plt.subplots(figsize=(8,4))

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

    st.markdown("---")

    st.subheader("💡 Interpretasi Keputusan")

    st.info(
        "Semakin tinggi nilai Vi maka semakin tinggi prioritas "
        "produk untuk memperoleh program diskon. Nilai tinggi "
        "menunjukkan kombinasi stok yang besar dan masa "
        "kedaluwarsa yang semakin dekat."
    )
