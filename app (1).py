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
    
    if "tarih" in df_price.columns:
        df_price["tarih"] = pd.to_datetime(df_price["tarih"], errors="coerce")
        df_price = df_price.sort_values("tarih").groupby("fon_kodu").tail(1)
    
    # 4. Dağılım dosyası - en son tarihi al
    df_alloc = pd.read_excel(alloc_file)
    df_alloc = clean_columns(df_alloc)
    
    if "Fon Kodu" in df_alloc.columns:
        df_alloc = df_alloc.rename(columns={"Fon Kodu": "fon_kodu", "Fon Adı": "fon_adi_alloc", "Tarih": "tarih_alloc"})
    
    if "tarih_alloc" in df_alloc.columns:
        df_alloc["tarih_alloc"] = pd.to_datetime(df_alloc["tarih_alloc"], errors="coerce")
        df_alloc = df_alloc.sort_values("tarih_alloc").groupby("fon_kodu").tail(1)
    
    # Birleştirme
    df = df_ret.copy()
    
    size_cols = [c for c in ["fon_kodu", "son_buyukluk", "buyukluk_degisim", "pay_degisim", "getiri_kisa"] if c in df_size.columns]
    if size_cols:
        df = df.merge(df_size[size_cols], on="fon_kodu", how="left")
    
    price_cols = [c for c in ["fon_kodu", "fiyat", "kisi_sayisi", "fon_toplam_deger"] if c in df_price.columns]
    if price_cols:
        df = df.merge(df_price[price_cols], on="fon_kodu", how="left")
    
    alloc_important = ["fon_kodu"]
    for col in df_alloc.columns:
        if any(x in str(col).lower() for x in ["hisse", "altın", "devlet", "döviz", "mevduat", "yabancı", "repo", "kıymetli"]):
            alloc_important.append(col)
    alloc_important = list(dict.fromkeys(alloc_important))
    if len(alloc_important) > 1:
        df = df.merge(df_alloc[alloc_important], on="fon_kodu", how="left")
    
    if "risk" in df.columns:
        df["risk"] = pd.to_numeric(df["risk"], errors="coerce")
    
    getiri_cols = [c for c in df.columns if c.startswith("getiri_")]
    for col in getiri_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "son_buyukluk" in df.columns:
        df["son_buyukluk"] = pd.to_numeric(df["son_buyukluk"], errors="coerce")
    if "buyukluk_degisim" in df.columns:
        df["buyukluk_degisim"] = pd.to_numeric(df["buyukluk_degisim"], errors="coerce")
    
    return df


def calculate_risk_adjusted_score(df, getiri_col, risk_free=0.35):
    """
    Basit risk-ayarlı skor hesapla.
    Skor = (Getiri - risksiz faiz) / Risk
    Risk 0 veya NaN ise düşük skor ver.
    risksiz faiz varsayılan ~%35 (yaklaşık para piyasası 1 yıllık seviye, ayarlanabilir)
    """
    df = df.copy()
    
    if getiri_col not in df.columns or "risk" not in df.columns:
        df["skor"] = 0
        return df
    
    getiri = df[getiri_col].fillna(0)
    risk = df["risk"].fillna(5)  # eksik risk varsa orta kabul et
    
    # Risk 0 veya çok düşükse payda sorun olmasın
    risk = risk.replace(0, 0.5)
    
    # Yıllık getiri yaklaşık ölçekleme (basit)
    # 1 aylık getiriyi kabaca yıllıklaştırmıyoruz, skor karşılaştırmalı olduğu için ham kullanıyoruz
    df["skor"] = (getiri - risk_free * 0.1) / risk   # risksiz kısmı dönemine göre yumuşatıldı
    
    # Negatif getiri + yüksek risk → daha kötü skor
    df.loc[getiri < 0, "skor"] = df.loc[getiri < 0, "skor"] * 1.5
    
    return df


def filter_by_liquidity(df, min_buyukluk=50_000_000, max_kuculme=-0.40):
    """
    Büyüklük ve likidite filtresi.
    - Minimum fon büyüklüğü (varsayılan 50 milyon TL)
    - Aşırı küçülen fonları ele (varsayılan %40'tan fazla küçülenler)
    """
    df = df.copy()
    
    if "son_buyukluk" in df.columns:
        df = df[(df["son_buyukluk"].isna()) | (df["son_buyukluk"] >= min_buyukluk)]
    
    if "buyukluk_degisim" in df.columns:
        # buyukluk_degisim genellikle ondalık (0.05 = %5) veya yüzde olabilir
        # Dosyada -0.0935 şeklinde geldiği için ondalık kabul ediyoruz
        df = df[(df["buyukluk_degisim"].isna()) | (df["buyukluk_degisim"] >= max_kuculme)]
    
    return df


def select_diversified_portfolio(df, n_funds=5, max_per_category=2, min_categories=3):
    """
    Farklı kategorilerden zorunlu çeşitlendirme ile fon seç.
    Skora göre sıralı listeden kategori kısıtlarını uygulayarak seçer.
    """
    if df.empty or "skor" not in df.columns:
        return df.head(0)
    
    df_sorted = df.sort_values("skor", ascending=False).copy()
    
    selected = []
    category_count = {}
    
    for _, row in df_sorted.iterrows():
        kat = row.get("kategori", "Bilinmeyen")
        if pd.isna(kat):
            kat = "Bilinmeyen"
        
        current = category_count.get(kat, 0)
        if current >= max_per_category:
            continue
        
        selected.append(row)
        category_count[kat] = current + 1
        
        if len(selected) >= n_funds:
            break
    
    result = pd.DataFrame(selected)
    
    # Yeterli kategori yoksa uyar ama yine de döndür
    return result


def calculate_portfolio_metrics(portfolio_df, weights):
    """Portföy metriklerini hesapla"""
    if portfolio_df.empty or not weights:
        return {}
    
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        return {}
    weights = weights / weights.sum()
    
    metrics = {}
    
    for col in ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y", "getiri_ybb", "skor"]:
        if col in portfolio_df.columns:
            vals = portfolio_df[col].fillna(0).values
            metrics[col] = np.sum(vals * weights)
    
    if "risk" in portfolio_df.columns:
        metrics["ortalama_risk"] = np.average(portfolio_df["risk"].fillna(3), weights=weights)
    
    return metrics


def suggest_rebalancing(current_portfolio, df, getiri_col="getiri_1y", top_n=5):
    """
    Mevcut portföye göre sat-al önerisi.
    - Mevcut fonların skorunu hesapla
    - Daha yüksek skorlu alternatifler öner
    """
    if not current_portfolio or df is None or df.empty:
        return None, None
    
    df = calculate_risk_adjusted_score(df, getiri_col)
    
    current_kods = list(current_portfolio.keys())
    current_df = df[df["fon_kodu"].isin(current_kods)].copy()
    
    if current_df.empty:
        return None, None
    
    # Mevcut portföy ortalama skoru
    weights = [current_portfolio.get(k, 0) for k in current_df["fon_kodu"]]
    current_metrics = calculate_portfolio_metrics(current_df, weights)
    current_avg_score = current_metrics.get("skor", 0)
    
    # Portföyde olmayan, daha yüksek skorlu fonlar
    candidates = df[~df["fon_kodu"].isin(current_kods)].copy()
    candidates = candidates[candidates["skor"] > current_avg_score * 0.9]  # en az mevcut ortalamanın %90'ı
    candidates = candidates.sort_values("skor", ascending=False).head(top_n)
    
    # Satılması önerilenler (en düşük skorlu mevcut fonlar)
    current_df = current_df.sort_values("skor", ascending=True)
    to_sell = current_df.head(min(2, len(current_df)))
    
    return to_sell, candidates


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
    st.session_state.portfolio = {}

# ==================== SAYFALAR ====================

# ----- ANA SAYFA -----
if page == "🏠 Ana Sayfa":
    st.title("📊 TEFAS Portföy Yöneticisi")
    st.markdown("""
    Bu uygulama ile TEFAS’tan indirdiğin fon verilerini analiz edebilir,  
    kendi portföyünü oluşturabilir ve **risk-ayarlı skor** + çeşitlendirme bazlı öneriler alabilirsin.
    
    ### Nasıl Kullanılır?
    1. **Veri Yükle** sekmesinden 4 Excel dosyasını yükle
    2. **Portföyüm** sekmesinden fonlarını ekle
    3. **Analiz** sekmesinde filtrele ve karşılaştır
    4. **Öneri Motoru** ile gelişmiş portföy önerileri al
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
            
            if st.button("💾 Ağırlıkları Kaydet"):
                new_port = {}
                for _, row in edited.iterrows():
                    new_port[row["Fon Kodu"]] = row["Ağırlık %"]
                st.session_state.portfolio = new_port
                st.success("Kaydedildi!")
                st.rerun()
            
            to_remove = st.multiselect("Silinecek fonlar", list(st.session_state.portfolio.keys()))
            if st.button("🗑️ Seçilenleri Sil") and to_remove:
                for kod in to_remove:
                    st.session_state.portfolio.pop(kod, None)
                st.rerun()
            
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
            
            fig = px.pie(port_df, values="Ağırlık %", names="Fon Kodu", title="Portföy Dağılımı")
            st.plotly_chart(fig, use_container_width=True)
            
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
        
        st.sidebar.markdown("### Filtreler")
        
        if "risk" in df.columns:
            risk_min, risk_max = st.sidebar.slider("Risk Aralığı", 1, 7, (1, 7))
            df = df[(df["risk"] >= risk_min) & (df["risk"] <= risk_max) | df["risk"].isna()]
        
        if "kategori" in df.columns:
            kategoriler = ["Tümü"] + sorted(df["kategori"].dropna().unique().tolist())
            secilen_kat = st.sidebar.selectbox("Kategori", kategoriler)
            if secilen_kat != "Tümü":
                df = df[df["kategori"] == secilen_kat]
        
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
        
        min_getiri = st.sidebar.number_input("Min. Getiri (%)", value=-100.0, step=5.0)
        if getiri_col in df.columns:
            df = df[df[getiri_col].fillna(-999) * 100 >= min_getiri]
        
        # Likidite filtresi
        min_buyukluk = st.sidebar.number_input("Min. Fon Büyüklüğü (Milyon TL)", value=0, step=10) * 1_000_000
        if "son_buyukluk" in df.columns and min_buyukluk > 0:
            df = df[(df["son_buyukluk"].isna()) | (df["son_buyukluk"] >= min_buyukluk)]
        
        st.subheader(f"Filtrelenmiş Fonlar ({len(df)} adet)")
        
        # Skor ekle
        df = calculate_risk_adjusted_score(df, getiri_col)
        
        if getiri_col in df.columns:
            df_sorted = df.sort_values("skor", ascending=False)
        else:
            df_sorted = df
        
        show_cols = ["fon_kodu", "fon_adi", "kategori", "risk", getiri_col, "skor"]
        if "getiri_1y" in df.columns and getiri_col != "getiri_1y":
            show_cols.append("getiri_1y")
        if "son_buyukluk" in df.columns:
            show_cols.append("son_buyukluk")
        
        show_cols = [c for c in show_cols if c in df_sorted.columns]
        
        st.dataframe(
            df_sorted[show_cols].head(50).style.format({
                getiri_col: lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                "getiri_1y": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                "skor": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                "son_buyukluk": lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
            }),
            use_container_width=True
        )
        
        st.subheader("Risk - Getiri Matrisi")
        if "risk" in df.columns and getiri_col in df.columns:
            plot_df = df.dropna(subset=["risk", getiri_col])
            fig = px.scatter(
                plot_df,
                x="risk",
                y=getiri_col,
                hover_name="fon_adi",
                hover_data=["fon_kodu", "kategori", "skor"],
                color="kategori" if "kategori" in plot_df.columns else None,
                size="skor" if "skor" in plot_df.columns else None,
                title=f"Risk vs {secilen_donem} Getiri (boyut = risk-ayarlı skor)",
                labels={getiri_col: f"{secilen_donem} Getiri", "risk": "Risk Değeri"}
            )
            fig.update_traces(marker=dict(opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
        
        if "kategori" in df.columns and getiri_col in df.columns:
            st.subheader("Kategori Bazlı Ortalama Performans")
            kat_perf = df.groupby("kategori")[getiri_col].mean().sort_values(ascending=False).reset_index()
            kat_perf.columns = ["Kategori", "Ortalama Getiri"]
            fig3 = px.bar(kat_perf, x="Kategori", y="Ortalama Getiri", title=f"Kategori Ortalama {secilen_donem}")
            st.plotly_chart(fig3, use_container_width=True)

# ----- ÖNERİ MOTORU -----
elif page == "🤖 Öneri Motoru":
    st.title("🤖 Portföy Öneri Motoru (Gelişmiş)")
    
    if st.session_state.df is None:
        st.warning("Önce **Veri Yükle** sekmesinden verileri yükle.")
    else:
        df = st.session_state.df.copy()
        
        st.markdown("""
        **Yeni özellikler:**
        - Risk-ayarlı skor (Getiri / Risk)
        - Farklı kategorilerden zorunlu çeşitlendirme
        - Büyüklük / likidite filtresi
        - Mevcut portföye göre sat-al önerisi
        """)
        
        tab1, tab2 = st.tabs(["🎯 Yeni Portföy Öner", "🔄 Mevcut Portföyü İyileştir"])
        
        # ========== TAB 1: Yeni Portföy ==========
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_pref = st.select_slider(
                    "Risk Toleransın",
                    options=[1, 2, 3, 4, 5, 6, 7],
                    value=4,
                    help="1 = Çok düşük risk, 7 = Yüksek risk"
                )
            
            with col2:
                donem = st.selectbox(
                    "Öncelikli Dönem",
                    ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y", "getiri_ybb"],
                    index=3,
                    format_func=lambda x: {
                        "getiri_1a": "1 Ay", "getiri_3a": "3 Ay", "getiri_6a": "6 Ay",
                        "getiri_1y": "1 Yıl", "getiri_ybb": "Yılbaşından"
                    }[x]
                )
            
            with col3:
                fon_sayisi = st.slider("Kaç fon önerilsin?", 3, 10, 5)
            
            # Likidite ayarları
            with st.expander("⚙️ Gelişmiş Filtreler"):
                min_buyukluk_m = st.number_input("Minimum Fon Büyüklüğü (Milyon TL)", value=50, step=10)
                max_kuculme = st.slider("Maksimum Küçülme Toleransı (%)", -100, 0, -40, help="Bu orandan fazla küçülen fonlar elenir")
                max_per_cat = st.slider("Aynı kategoriden maksimum fon", 1, 3, 2)
                min_cats = st.slider("Minimum farklı kategori", 2, 5, 3)
            
            if "kategori" in df.columns:
                st.markdown("**Kategori Tercihi** (boş bırakırsan tümü)")
                secilen_kategoriler = st.multiselect(
                    "Kategoriler",
                    options=sorted(df["kategori"].dropna().unique()),
                    default=[]
                )
            else:
                secilen_kategoriler = []
            
            if st.button("🎯 Öneri Oluştur", type="primary", key="yeni_oneri"):
                filtered = df.copy()
                
                # 1. Risk filtresi (±1)
                risk_low = max(1, risk_pref - 1)
                risk_high = min(7, risk_pref + 1)
                if "risk" in filtered.columns:
                    filtered = filtered[
                        ((filtered["risk"] >= risk_low) & (filtered["risk"] <= risk_high)) |
                        (filtered["risk"].isna())
                    ]
                
                # 2. Kategori filtresi
                if secilen_kategoriler:
                    filtered = filtered[filtered["kategori"].isin(secilen_kategoriler)]
                
                # 3. Likidite filtresi
                filtered = filter_by_liquidity(
                    filtered,
                    min_buyukluk=min_buyukluk_m * 1_000_000,
                    max_kuculme=max_kuculme / 100.0
                )
                
                # 4. Risk-ayarlı skor hesapla
                filtered = calculate_risk_adjusted_score(filtered, donem)
                
                # 5. Çeşitlendirilmiş seçim
                oneriler = select_diversified_portfolio(
                    filtered,
                    n_funds=fon_sayisi,
                    max_per_category=max_per_cat,
                    min_categories=min_cats
                )
                
                if oneriler.empty:
                    st.warning("Kriterlere uygun fon bulunamadı. Filtreleri gevşet.")
                else:
                    n_kat = oneriler["kategori"].nunique() if "kategori" in oneriler.columns else 0
                    st.success(f"**{len(oneriler)}** fon önerildi | **{n_kat}** farklı kategori | Risk ≈ {risk_pref}")
                    
                    esit_agirlik = 100 / len(oneriler)
                    
                    oneri_data = []
                    for _, r in oneriler.iterrows():
                        oneri_data.append({
                            "Fon Kodu": r["fon_kodu"],
                            "Fon Adı": r.get("fon_adi", ""),
                            "Önerilen Ağırlık %": round(esit_agirlik, 1),
                            "Risk": r.get("risk", "-"),
                            "Skor": r.get("skor", np.nan),
                            "Dönem Getiri": r.get(donem, np.nan),
                            "1 Yıl Getiri": r.get("getiri_1y", np.nan),
                            "Kategori": r.get("kategori", ""),
                            "Büyüklük": r.get("son_buyukluk", np.nan)
                        })
                    
                    oneri_df = pd.DataFrame(oneri_data)
                    
                    st.dataframe(
                        oneri_df.style.format({
                            "Skor": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                            "Dönem Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                            "1 Yıl Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                            "Büyüklük": lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
                        }),
                        use_container_width=True
                    )
                    
                    if st.button("📥 Bu öneriyi Portföyüme Ekle", key="ekle_yeni"):
                        for _, row in oneri_df.iterrows():
                            st.session_state.portfolio[row["Fon Kodu"]] = row["Önerilen Ağırlık %"]
                        st.success("Portföye eklendi!")
                        st.rerun()
                    
                    fig = px.pie(oneri_df, values="Önerilen Ağırlık %", names="Fon Kodu", title="Önerilen Portföy Dağılımı")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Kategori dağılımı
                    if "Kategori" in oneri_df.columns:
                        kat_df = oneri_df.groupby("Kategori")["Önerilen Ağırlık %"].sum().reset_index()
                        fig_kat = px.pie(kat_df, values="Önerilen Ağırlık %", names="Kategori", title="Kategori Çeşitliliği")
                        st.plotly_chart(fig_kat, use_container_width=True)
                    
                    st.subheader("Önerilen Portföy Tahmini")
                    weights = [esit_agirlik] * len(oneriler)
                    metrics = calculate_portfolio_metrics(oneriler, weights)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if donem in metrics:
                            st.metric("Beklenen Getiri", f"%{metrics[donem]*100:.2f}")
                    with c2:
                        if "ortalama_risk" in metrics:
                            st.metric("Ortalama Risk", f"{metrics['ortalama_risk']:.1f}")
                    with c3:
                        if "getiri_1y" in metrics:
                            st.metric("1 Yıllık Tahmini", f"%{metrics['getiri_1y']*100:.2f}")
                    with c4:
                        if "skor" in metrics:
                            st.metric("Ort. Skor", f"{metrics['skor']:.3f}")
        
        # ========== TAB 2: Mevcut Portföyü İyileştir ==========
        with tab2:
            st.markdown("Mevcut portföyündeki zayıf fonları tespit edip daha iyi alternatifler önerir.")
            
            if not st.session_state.portfolio:
                st.info("Önce **Portföyüm** sekmesinden fon ekle. Sonra burada iyileştirme önerisi alabilirsin.")
            else:
                donem2 = st.selectbox(
                    "Karşılaştırma Dönemi",
                    ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y", "getiri_ybb"],
                    index=3,
                    format_func=lambda x: {
                        "getiri_1a": "1 Ay", "getiri_3a": "3 Ay", "getiri_6a": "6 Ay",
                        "getiri_1y": "1 Yıl", "getiri_ybb": "Yılbaşından"
                    }[x],
                    key="donem_rebal"
                )
                
                if st.button("🔄 İyileştirme Önerisi Getir", type="primary"):
                    to_sell, candidates = suggest_rebalancing(
                        st.session_state.portfolio,
                        df,
                        getiri_col=donem2,
                        top_n=5
                    )
                    
                    if to_sell is None:
                        st.warning("Analiz yapılamadı.")
                    else:
                        st.subheader("📉 Satılması Düşünülebilecek Fonlar (düşük skor)")
                        if to_sell.empty:
                            st.success("Portföyündeki fonlar genel olarak iyi skorlara sahip.")
                        else:
                            sell_data = []
                            for _, r in to_sell.iterrows():
                                sell_data.append({
                                    "Fon Kodu": r["fon_kodu"],
                                    "Fon Adı": r.get("fon_adi", ""),
                                    "Mevcut Ağırlık %": st.session_state.portfolio.get(r["fon_kodu"], 0),
                                    "Risk": r.get("risk", "-"),
                                    "Skor": r.get("skor", np.nan),
                                    "Dönem Getiri": r.get(donem2, np.nan),
                                    "Kategori": r.get("kategori", "")
                                })
                            st.dataframe(
                                pd.DataFrame(sell_data).style.format({
                                    "Skor": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                                    "Dönem Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-"
                                }),
                                use_container_width=True
                            )
                        
                        st.subheader("📈 Alınması Düşünülebilecek Alternatifler (yüksek skor)")
                        if candidates is None or candidates.empty:
                            st.info("Mevcut portföyden belirgin şekilde daha iyi alternatif bulunamadı.")
                        else:
                            buy_data = []
                            for _, r in candidates.iterrows():
                                buy_data.append({
                                    "Fon Kodu": r["fon_kodu"],
                                    "Fon Adı": r.get("fon_adi", ""),
                                    "Risk": r.get("risk", "-"),
                                    "Skor": r.get("skor", np.nan),
                                    "Dönem Getiri": r.get(donem2, np.nan),
                                    "1 Yıl Getiri": r.get("getiri_1y", np.nan),
                                    "Kategori": r.get("kategori", ""),
                                    "Büyüklük": r.get("son_buyukluk", np.nan)
                                })
                            st.dataframe(
                                pd.DataFrame(buy_data).style.format({
                                    "Skor": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                                    "Dönem Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                                    "1 Yıl Getiri": lambda x: f"%{x*100:.2f}" if pd.notna(x) else "-",
                                    "Büyüklük": lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
                                }),
                                use_container_width=True
                            )
                        
                        st.info("💡 Bu öneriler otomatik al-sat yapmaz. Kendi değerlendirmeni yaparak karar ver.")

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
    
    ### Risk-Ayarlı Skor Nasıl Hesaplanır?
    ```
    Skor = (Getiri - risksiz faiz payı) / Risk
    ```
    - Yüksek getiri + düşük risk → yüksek skor
    - Negatif getiri + yüksek risk → daha da düşük skor
    
    ### Risk Değerleri
    - **1-2**: Çok düşük risk (para piyasası, kısa vadeli borçlanma)
    - **3-4**: Düşük-orta risk
    - **5**: Orta risk
    - **6-7**: Yüksek risk (hisse yoğun, serbest, yabancı)
    
    ### Çeşitlendirme Kuralları
    - Aynı kategoriden maksimum 2 fon
    - Mümkün olduğunca farklı kategorilerden seçim
    
    ### İpuçları
    - Portföyünde 4-8 fon tutmak genelde yeterlidir.
    - Farklı kategorilerden fon seçerek çeşitlendir.
    - Sadece geçmiş getirilere bakma, risk ve büyüklük değişimini de incele.
    """)
    
    st.markdown("---")
    st.caption("Bu uygulama yatırım tavsiyesi değildir. Kendi araştırmanı yap.")
