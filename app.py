import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# Sayfa ayarları
st.set_page_config(
    page_title="TEFAS Portföy Yöneticisi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== YARDIMCI FONKSİYONLAR ====================

def clean_columns(df):
    """Sütun isimlerini temizle ve standartlaştır"""
    df.columns = df.columns.str.strip()
    return df

def load_and_prepare_data(returns_file, size_file, price_file, alloc_file):
    """4 dosyayı yükle ve birleştir"""
    
    # 1. Getiri ve Risk dosyası
    df_ret = pd.read_excel(returns_file)
    df_ret = clean_columns(df_ret)
    
    # Sütun isimlerini standartlaştır
    rename_ret = {
        "Fon Kodu": "fon_kodu",
        "Fon Adı": "fon_adi",
        "Şemsiye Fon Türü": "kategori",
        "Fonun Risk Değeri": "risk",
        "1 Ay (%)": "getiri_1a",
        "3 Ay (%)": "getiri_3a",
        "6 Ay (%)": "getiri_6a",
        "Yılbaşından İtibaren (%)": "getiri_ybb",
        "1 Yıl (%)": "getiri_1y",
        "3 Yıl (%)": "getiri_3y",
        "5 Yıl (%)": "getiri_5y"
    }
    df_ret = df_ret.rename(columns={k: v for k, v in rename_ret.items() if k in df_ret.columns})
    
    # 2. Büyüklük dosyası
    df_size = pd.read_excel(size_file)
    df_size = clean_columns(df_size)
    rename_size = {
        "Fon Kodu": "fon_kodu",
        "Fon Adı": "fon_adi_size",
        "Şemsiye Fon Türü": "kategori_size",
        "İlk Portföy Büyüklüğü": "ilk_buyukluk",
        "Son Portföy Büyüklüğü": "son_buyukluk",
        "Portföy Büyüklüğü Değişimi (%)": "buyukluk_degisim",
        "Tedavüldeki İlk Pay Adedi": "ilk_pay",
        "Tedavüldeki Son Pay Adedi": "son_pay",
        "Pay Adedi Değişimi (%)": "pay_degisim",
        "Getiri Oranı (%)": "getiri_kisa"
    }
    df_size = df_size.rename(columns={k: v for k, v in rename_size.items() if k in df_size.columns})
    
    # 3. Genel (Fiyat) dosyası - en son tarihi al
    df_price = pd.read_excel(price_file)
    df_price = clean_columns(df_price)
    rename_price = {
        "Fon Kodu": "fon_kodu",
        "Fon Adı": "fon_adi_price",
        "Tarih": "tarih",
        "Fiyat": "fiyat",
        "Tedavüldeki Pay Sayısı": "pay_sayisi",
        "Kişi Sayısı": "kisi_sayisi",
        "Fon Toplam Değer": "fon_toplam_deger"
    }
    df_price = df_price.rename(columns={k: v for k, v in rename_price.items() if k in df_price.columns})
    
    # En son tarihi al
    if "tarih" in df_price.columns:
        df_price["tarih"] = pd.to_datetime(df_price["tarih"], errors="coerce")
        df_price = df_price.sort_values("tarih").groupby("fon_kodu").tail(1)
    
    # 4. Dağılım dosyası - en son tarihi al
    df_alloc = pd.read_excel(alloc_file)
    df_alloc = clean_columns(df_alloc)
    
    # Ana sütunları koru, diğerlerini yüzde olarak bırak
    if "Fon Kodu" in df_alloc.columns:
        df_alloc = df_alloc.rename(columns={"Fon Kodu": "fon_kodu", "Fon Adı": "fon_adi_alloc", "Tarih": "tarih_alloc"})
    
    if "tarih_alloc" in df_alloc.columns:
        df_alloc["tarih_alloc"] = pd.to_datetime(df_alloc["tarih_alloc"], errors="coerce")
        df_alloc = df_alloc.sort_values("tarih_alloc").groupby("fon_kodu").tail(1)
    
    # Birleştirme
    df = df_ret.copy()
    
    # Size ekle
    size_cols = [c for c in ["fon_kodu", "son_buyukluk", "buyukluk_degisim", "pay_degisim", "getiri_kisa"] if c in df_size.columns]
    if size_cols:
        df = df.merge(df_size[size_cols], on="fon_kodu", how="left")
    
    # Price ekle
    price_cols = [c for c in ["fon_kodu", "fiyat", "kisi_sayisi", "fon_toplam_deger"] if c in df_price.columns]
    if price_cols:
        df = df.merge(df_price[price_cols], on="fon_kodu", how="left")
    
    # Alloc ekle (sadece hisse ve önemli olanlar)
    alloc_important = ["fon_kodu"]
    for col in df_alloc.columns:
        if any(x in col.lower() for x in ["hisse", "altın", "devlet", "döviz", "mevduat", "yabancı", "repo", "kıymetli"]):
            alloc_important.append(col)
    
    alloc_important = list(dict.fromkeys(alloc_important))  # unique
    if len(alloc_important) > 1:
        df = df.merge(df_alloc[alloc_important], on="fon_kodu", how="left")
    
    # Risk sayıya çevir
    if "risk" in df.columns:
        df["risk"] = pd.to_numeric(df["risk"], errors="coerce")
    
    # Getiri sütunlarını sayıya çevir
    getiri_cols = [c for c in df.columns if c.startswith("getiri_")]
    for col in getiri_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def format_percent(x):
    if pd.isna(x):
        return "-"
    return f"%{x*100:.2f}" if abs(x) < 10 else f"%{x:.2f}"

def calculate_portfolio_metrics(portfolio_df, weights):
    """Portföy metriklerini hesapla"""
    if portfolio_df.empty or not weights:
        return {}
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize
    
    metrics = {}
    
    for col in ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y", "getiri_ybb"]:
        if col in portfolio_df.columns:
            vals = portfolio_df[col].fillna(0).values
            metrics[col] = np.sum(vals * weights)
    
    if "risk" in portfolio_df.columns:
        metrics["ortalama_risk"] = np.average(portfolio_df["risk"].fillna(3), weights=weights)
    
    return metrics

# ==================== SIDEBAR ====================

st.sidebar.title("📊 TEFAS Portföy")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Menü",
    ["🏠 Ana Sayfa", "📂 Veri Yükle", "💼 Portföyüm", "📈 Analiz & Karşılaştırma", "🤖 Öneri Motoru", "ℹ️ Yardım"]
)

st.sidebar.markdown("---")
st.sidebar.info("Verilerini yükledikten sonra portföy oluşturup analiz yapabilirsin.")

# ==================== SESSION STATE ====================

if "df" not in st.session_state:
    st.session_state.df = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}  # {fon_kodu: weight}

# ==================== SAYFALAR ====================

# ----- ANA SAYFA -----
if page == "🏠 Ana Sayfa":
    st.title("📊 TEFAS Portföy Yöneticisi")
    st.markdown("""
    Bu uygulama ile TEFAS’tan indirdiğin fon verilerini analiz edebilir,  
    kendi portföyünü oluşturabilir ve risk-getiri bazlı öneriler alabilirsin.
    
    ### Nasıl Kullanılır?
    1. **Veri Yükle** sekmesinden 4 Excel dosyasını yükle
    2. **Portföyüm** sekmesinden fonlarını ekle
    3. **Analiz** sekmesinde filtrele ve karşılaştır
    4. **Öneri Motoru** ile yeni portföy önerileri al
    """)
    
    if st.session_state.df is not None:
        st.success(f"✅ Veri yüklendi: **{len(st.session_state.df)}** fon")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toplam Fon", len(st.session_state.df))
        with col2:
            risk_avg = st.session_state.df["risk"].mean() if "risk" in st.session_state.df.columns else 0
            st.metric("Ort. Risk", f"{risk_avg:.1f}" if risk_avg else "-")
        with col3:
            if "getiri_1y" in st.session_state.df.columns:
                st.metric("Ort. 1Y Getiri", f"%{st.session_state.df['getiri_1y'].mean()*100:.1f}")
        with col4:
            st.metric("Portföydeki Fon", len(st.session_state.portfolio))
    else:
        st.warning("Henüz veri yüklenmedi. Sol menüden **Veri Yükle** sekmesine git.")

# ----- VERİ YÜKLE -----
elif page == "📂 Veri Yükle":
    st.title("📂 Veri Yükleme")
    st.markdown("TEFAS’tan indirdiğin 4 Excel dosyasını buraya yükle.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        returns_file = st.file_uploader("1️⃣ Getiri & Risk Dosyası (Menkul Kıymet...)", type=["xlsx", "xls"], key="ret")
        size_file = st.file_uploader("2️⃣ Büyüklük Dosyası", type=["xlsx", "xls"], key="size")
    
    with col2:
        price_file = st.file_uploader("3️⃣ Genel / Fiyat Dosyası", type=["xlsx", "xls"], key="price")
        alloc_file = st.file_uploader("4️⃣ Dağılım Dosyası", type=["xlsx", "xls"], key="alloc")
    
    if st.button("🔄 Verileri Birleştir ve Yükle", type="primary"):
        if returns_file and size_file and price_file and alloc_file:
            with st.spinner("Veriler işleniyor..."):
                try:
                    df = load_and_prepare_data(returns_file, size_file, price_file, alloc_file)
                    st.session_state.df = df
                    st.success(f"✅ Başarıyla yüklendi! **{len(df)}** fon hazır.")
                    
                    st.subheader("Önizleme (ilk 10 satır)")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.subheader("Sütunlar")
                    st.write(list(df.columns))
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")
        else:
            st.warning("Lütfen 4 dosyayı da yükle.")
    
    # Demo için örnek veri yükleme butonu (geliştirme amaçlı)
    st.markdown("---")
    st.caption("Geliştirme notu: Dosyalar zaten /attachments klasöründe mevcutsa otomatik yükleyebilirsin.")

# ----- PORTFÖYÜM -----
elif page == "💼 Portföyüm":
    st.title("💼 Portföyüm")
    
    if st.session_state.df is None:
        st.warning("Önce **Veri Yükle** sekmesinden verileri yükle.")
    else:
        df = st.session_state.df
        
        st.subheader("Fon Ekle")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Fon arama
            search = st.text_input("Fon kodu veya adı ile ara", placeholder="Örn: AAL veya Altın")
            if search:
                mask = (
                    df["fon_kodu"].str.contains(search, case=False, na=False) |
                    df["fon_adi"].str.contains(search, case=False, na=False)
                )
                options = df[mask]["fon_kodu"] + " - " + df[mask]["fon_adi"]
            else:
                options = df["fon_kodu"] + " - " + df["fon_adi"]
            
            selected = st.selectbox("Fon seç", options)
            selected_kod = selected.split(" - ")[0] if selected else None
        
        with col2:
            weight = st.number_input("Ağırlık (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        
        with col3:
            st.write("")
            st.write("")
            if st.button("➕ Ekle", type="primary") and selected_kod:
                st.session_state.portfolio[selected_kod] = weight
                st.success(f"{selected_kod} eklendi!")
                st.rerun()
        
        st.markdown("---")
        st.subheader("Mevcut Portföy")
        
        if st.session_state.portfolio:
            port_data = []
            for kod, w in st.session_state.portfolio.items():
                row = df[df["fon_kodu"] == kod]
                if not row.empty:
                    r = row.iloc[0]
                    port_data.append({
                        "Fon Kodu": kod,
                        "Fon Adı": r.get("fon_adi", ""),
                        "Ağırlık %": w,
                        "Risk": r.get("risk", "-"),
                        "1 Ay": r.get("getiri_1a", np.nan),
                        "3 Ay": r.get("getiri_3a", np.nan),
                        "1 Yıl": r.get("getiri_1y", np.nan),
                        "Kategori": r.get("kategori", "")
                    })
            
            port_df = pd.DataFrame(port_data)
            
            # Ağırlık düzenleme
            edited = st.data_editor(
                port_df,
                column_config={
                    "Ağırlık %": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
                    "1 Ay": st.column_config.NumberColumn(format="%.2f%%"),
                    "3 Ay": st.column_config.NumberColumn(format="%.2f%%"),
                    "1 Yıl": st.column_config.NumberColumn(format="%.2f%%"),
                },
                use_container_width=True,
                num_rows="fixed",
                key="port_editor"
            )
            
            # Güncelle
            if st.button("💾 Ağırlıkları Kaydet"):
                new_port = {}
                for _, row in edited.iterrows():
                    new_port[row["Fon Kodu"]] = row["Ağırlık %"]
                st.session_state.portfolio = new_port
                st.success("Kaydedildi!")
                st.rerun()
            
            # Silme
            to_remove = st.multiselect("Silinecek fonlar", list(st.session_state.portfolio.keys()))
            if st.button("🗑️ Seçilenleri Sil") and to_remove:
                for kod in to_remove:
                    st.session_state.portfolio.pop(kod, None)
                st.rerun()
            
            # Portföy metrikleri
            st.markdown("---")
            st.subheader("Portföy Özeti")
            
            weights = [st.session_state.portfolio[k] for k in port_df["Fon Kodu"]]
            metrics = calculate_portfolio_metrics(port_df, weights)
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                if "getiri_1a" in metrics:
                    st.metric("Portföy 1 Ay", f"%{metrics['getiri_1a']*100:.2f}")
            with m2:
                if "getiri_3a" in metrics:
                    st.metric("Portföy 3 Ay", f"%{metrics['getiri_3a']*100:.2f}")
            with m3:
                if "getiri_1y" in metrics:
                    st.metric("Portföy 1 Yıl", f"%{metrics['getiri_1y']*100:.2f}")
            with m4:
                if "ortalama_risk" in metrics:
                    st.metric("Ort. Risk", f"{metrics['ortalama_risk']:.1f}")
            
            # Pasta grafik
            fig = px.pie(port_df, values="Ağırlık %", names="Fon Kodu", title="Portföy Dağılımı")
            st.plotly_chart(fig, use_container_width=True)
            
            # Kategori dağılımı
            if "Kategori" in port_df.columns:
                kat = port_df.groupby("Kategori")["Ağırlık %"].sum().reset_index()
                fig2 = px.pie(kat, values="Ağırlık %", names="Kategori", title="Kategori Dağılımı")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Henüz portföyüne fon eklemedin.")

# ----- ANALİZ -----
elif page == "📈 Analiz & Karşılaştırma":
    st.title("📈 Analiz & Karşılaştırma")
    
    if st.session_state.df is None:
        st.warning("Önce **Veri Yükle** sekmesinden verileri yükle.")
    else:
        df = st.session_state.df.copy()
        
        # Filtreler
        st.sidebar.markdown("### Filtreler")
        
        # Risk filtresi
        if "risk" in df.columns:
            risk_min, risk_max = st.sidebar.slider("Risk Aralığı", 1, 7, (1, 7))
            df = df[(df["risk"] >= risk_min) & (df["risk"] <= risk_max) | df["risk"].isna()]
        
        # Kategori filtresi
        if "kategori" in df.columns:
            kategoriler = ["Tümü"] + sorted(df["kategori"].dropna().unique().tolist())
            secilen_kat = st.sidebar.selectbox("Kategori", kategoriler)
            if secilen_kat != "Tümü":
                df = df[df["kategori"] == secilen_kat]
        
        # Getiri dönemi
        getiri_secenek = {
            "1 Ay": "getiri_1a",
            "3 Ay": "getiri_3a",
            "6 Ay": "getiri_6a",
            "Yılbaşından": "getiri_ybb",
            "1 Yıl": "getiri_1y",
            "3 Yıl": "getiri_3y"
        }
        secilen_donem = st.sidebar.selectbox("Sıralama Dönemi", list(getiri_secenek.keys()), index=4)
        getiri_col = getiri_secenek[secilen_donem]
        
        # Minimum getiri
        min_getiri = st.sidebar.number_input("Min. Getiri (%)", value=-100.0, step=5.0)
        if getiri_col in df.columns:
            df = df[df[getiri_col].fillna(-999) * 100 >= min_getiri]
        
        st.subheader(f"Filtrelenmiş Fonlar ({len(df)} adet)")
        
        # Sıralama
        if getiri_col in df.columns:
            df_sorted = df.sort_values(getiri_col, ascending=False)
        else:
            df_sorted = df
        
        # Gösterilecek sütunlar
        show_cols = ["fon_kodu", "fon_adi", "kategori", "risk", getiri_col]
        if "getiri_1y" in df.columns and getiri_col != "getiri_1y":
            show_cols.append("getiri_1y")
        if "son_buyukluk" in df.columns:
            show_cols.append("son_buyukluk")
        
        show_cols = [c for c in show_cols if c in df_sorted.columns]
        
        st.dataframe(
            df_sorted[show_cols].head(50).style.format({
                getiri_col: lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                "getiri_1y": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                "son_buyukluk": lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
            }),
            use_container_width=True
        )
        
        # Risk-Getiri Scatter
        st.subheader("Risk - Getiri Matrisi")
        if "risk" in df.columns and getiri_col in df.columns:
            plot_df = df.dropna(subset=["risk", getiri_col])
            fig = px.scatter(
                plot_df,
                x="risk",
                y=getiri_col,
                hover_name="fon_adi",
                hover_data=["fon_kodu", "kategori"],
                color="kategori" if "kategori" in plot_df.columns else None,
                title=f"Risk vs {secilen_donem} Getiri",
                labels={getiri_col: f"{secilen_donem} Getiri", "risk": "Risk Değeri"}
            )
            fig.update_traces(marker=dict(size=10, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
        
        # Kategori bazlı ortalama
        if "kategori" in df.columns and getiri_col in df.columns:
            st.subheader("Kategori Bazlı Ortalama Performans")
            kat_perf = df.groupby("kategori")[getiri_col].mean().sort_values(ascending=False).reset_index()
            kat_perf.columns = ["Kategori", "Ortalama Getiri"]
            fig3 = px.bar(kat_perf, x="Kategori", y="Ortalama Getiri", title=f"Kategori Ortalama {secilen_donem}")
            st.plotly_chart(fig3, use_container_width=True)

# ----- ÖNERİ MOTORU -----
elif page == "🤖 Öneri Motoru":
    st.title("🤖 Portföy Öneri Motoru")
    
    if st.session_state.df is None:
        st.warning("Önce **Veri Yükle** sekmesinden verileri yükle.")
    else:
        df = st.session_state.df.copy()
        
        st.markdown("Risk toleransına ve tercihlerine göre otomatik portföy önerisi al.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_pref = st.select_slider(
                "Risk Toleransın",
                options=[1, 2, 3, 4, 5, 6, 7],
                value=4,
                help="1 = Çok düşük risk, 7 = Yüksek risk"
            )
        
        with col2:
            donem = st.selectbox("Öncelikli Dönem", ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y", "getiri_ybb"], index=3,
                                 format_func=lambda x: {"getiri_1a": "1 Ay", "getiri_3a": "3 Ay", "getiri_6a": "6 Ay",
                                                        "getiri_1y": "1 Yıl", "getiri_ybb": "Yılbaşından"}[x])
        
        with col3:
            fon_sayisi = st.slider("Kaç fon önerilsin?", 3, 10, 5)
        
        # Kategori tercihi
        if "kategori" in df.columns:
            st.markdown("**Kategori Tercihi** (boş bırakırsan tümü)")
            secilen_kategoriler = st.multiselect(
                "Kategoriler",
                options=sorted(df["kategori"].dropna().unique()),
                default=[]
            )
        
        if st.button("🎯 Öneri Oluştur", type="primary"):
            # Filtrele
            filtered = df.copy()
            
            # Risk aralığı: tercih ±1
            risk_low = max(1, risk_pref - 1)
            risk_high = min(7, risk_pref + 1)
            if "risk" in filtered.columns:
                filtered = filtered[(filtered["risk"] >= risk_low) & (filtered["risk"] <= risk_high) | filtered["risk"].isna()]
            
            if secilen_kategoriler:
                filtered = filtered[filtered["kategori"].isin(secilen_kategoriler)]
            
            # Getiriye göre sırala
            if donem in filtered.columns:
                filtered = filtered.dropna(subset=[donem])
                filtered = filtered.sort_values(donem, ascending=False)
            
            oneriler = filtered.head(fon_sayisi)
            
            if oneriler.empty:
                st.warning("Kriterlere uygun fon bulunamadı. Filtreleri gevşet.")
            else:
                st.success(f"**{len(oneriler)}** fon önerildi (Risk ≈ {risk_pref})")
                
                # Eşit ağırlıklı portföy
                esit_agirlik = 100 / len(oneriler)
                
                oneri_data = []
                for _, r in oneriler.iterrows():
                    oneri_data.append({
                        "Fon Kodu": r["fon_kodu"],
                        "Fon Adı": r.get("fon_adi", ""),
                        "Önerilen Ağırlık %": round(esit_agirlik, 1),
                        "Risk": r.get("risk", "-"),
                        "Seçilen Dönem Getiri": r.get(donem, np.nan),
                        "1 Yıl Getiri": r.get("getiri_1y", np.nan),
                        "Kategori": r.get("kategori", "")
                    })
                
                oneri_df = pd.DataFrame(oneri_data)
                
                st.dataframe(
                    oneri_df.style.format({
                        "Seçilen Dönem Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                        "1 Yıl Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-"
                    }),
                    use_container_width=True
                )
                
                # Portföye ekle butonu
                if st.button("📥 Bu öneriyi Portföyüme Ekle"):
                    for _, row in oneri_df.iterrows():
                        st.session_state.portfolio[row["Fon Kodu"]] = row["Önerilen Ağırlık %"]
                    st.success("Portföye eklendi! Portföyüm sekmesinden kontrol edebilirsin.")
                    st.rerun()
                
                # Pasta
                fig = px.pie(oneri_df, values="Önerilen Ağırlık %", names="Fon Kodu", title="Önerilen Portföy Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
                
                # Basit metrikler
                st.subheader("Önerilen Portföy Tahmini")
                weights = [esit_agirlik] * len(oneriler)
                metrics = calculate_portfolio_metrics(oneriler, weights)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if donem in metrics:
                        st.metric("Beklenen Getiri", f"%{metrics[donem]*100:.2f}")
                with c2:
                    if "ortalama_risk" in metrics:
                        st.metric("Ortalama Risk", f"{metrics['ortalama_risk']:.1f}")
                with c3:
                    if "getiri_1y" in metrics:
                        st.metric("1 Yıllık Tahmini", f"%{metrics['getiri_1y']*100:.2f}")

# ----- YARDIM -----
elif page == "ℹ️ Yardım":
    st.title("ℹ️ Yardım & Açıklamalar")
    
    st.markdown("""
    ### Dosya Formatları
    
    Uygulama şu 4 TEFAS Excel dosyasını bekler:
    
    1. **Getiri & Risk** → `Menkul_Kiymet_Yatirim_Fonlari_EXCEL_Tum_Veri_...xlsx`  
       Sütunlar: Fon Kodu, Fon Adı, Şemsiye Fon Türü, Risk, 1 Ay, 3 Ay, 6 Ay, YBB, 1Y, 3Y, 5Y
    
    2. **Büyüklük** → `Büyüklük (69).xlsx`  
       Portföy büyüklüğü değişimi, pay adedi değişimi
    
    3. **Genel / Fiyat** → `Genel (81).xlsx`  
       Tarih, Fiyat, Kişi Sayısı, Fon Toplam Değer
    
    4. **Dağılım** → `Dağılım (74).xlsx`  
       Varlık dağılımı yüzdeleri
    
    ### Risk Değerleri
    - **1-2**: Çok düşük risk (para piyasası, kısa vadeli borçlanma)
    - **3-4**: Düşük-orta risk
    - **5**: Orta risk
    - **6-7**: Yüksek risk (hisse yoğun, serbest, yabancı)
    
    ### İpuçları
    - Portföyünde 4-8 fon tutmak genelde yeterlidir.
    - Farklı kategorilerden fon seçerek çeşitlendir.
    - Sadece geçmiş getirilere bakma, risk ve büyüklük değişimini de incele.
    """)
    
    st.markdown("---")
    st.caption("Bu uygulama yatırım tavsiyesi değildir. Kendi araştırmanı yap.")
