import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TEFAS Portföy Yöneticisi - Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== YARDIMCI FONKSİYONLAR ====================

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def load_and_prepare_data(returns_file, size_file, price_file, alloc_file):
    df_ret = pd.read_excel(returns_file)
    df_ret = clean_columns(df_ret)
    rename_ret = {
        "Fon Kodu": "fon_kodu", "Fon Adı": "fon_adi", "Şemsiye Fon Türü": "kategori",
        "Fonun Risk Değeri": "risk", "1 Ay (%)": "getiri_1a", "3 Ay (%)": "getiri_3a",
        "6 Ay (%)": "getiri_6a", "Yılbaşından İtibaren (%)": "getiri_ybb",
        "1 Yıl (%)": "getiri_1y", "3 Yıl (%)": "getiri_3y", "5 Yıl (%)": "getiri_5y"
    }
    df_ret = df_ret.rename(columns={k: v for k, v in rename_ret.items() if k in df_ret.columns})

    df_size = pd.read_excel(size_file)
    df_size = clean_columns(df_size)
    rename_size = {
        "Fon Kodu": "fon_kodu", "Son Portföy Büyüklüğü": "son_buyukluk",
        "Portföy Büyüklüğü Değişimi (%)": "buyukluk_degisim",
        "Pay Adedi Değişimi (%)": "pay_degisim", "Getiri Oranı (%)": "getiri_kisa"
    }
    df_size = df_size.rename(columns={k: v for k, v in rename_size.items() if k in df_size.columns})

    df_price = pd.read_excel(price_file)
    df_price = clean_columns(df_price)
    rename_price = {
        "Fon Kodu": "fon_kodu", "Tarih": "tarih", "Fiyat": "fiyat",
        "Kişi Sayısı": "kisi_sayisi", "Fon Toplam Değer": "fon_toplam_deger"
    }
    df_price = df_price.rename(columns={k: v for k, v in rename_price.items() if k in df_price.columns})
    if "tarih" in df_price.columns:
        df_price["tarih"] = pd.to_datetime(df_price["tarih"], errors="coerce")
        df_price = df_price.sort_values("tarih").groupby("fon_kodu").tail(1)

    df_alloc = pd.read_excel(alloc_file)
    df_alloc = clean_columns(df_alloc)
    if "Fon Kodu" in df_alloc.columns:
        df_alloc = df_alloc.rename(columns={"Fon Kodu": "fon_kodu", "Tarih": "tarih_alloc"})
    if "tarih_alloc" in df_alloc.columns:
        df_alloc["tarih_alloc"] = pd.to_datetime(df_alloc["tarih_alloc"], errors="coerce")
        df_alloc = df_alloc.sort_values("tarih_alloc").groupby("fon_kodu").tail(1)

    df = df_ret.copy()
    size_cols = [c for c in ["fon_kodu", "son_buyukluk", "buyukluk_degisim", "pay_degisim"] if c in df_size.columns]
    if size_cols:
        df = df.merge(df_size[size_cols], on="fon_kodu", how="left")
    price_cols = [c for c in ["fon_kodu", "fiyat", "kisi_sayisi", "fon_toplam_deger"] if c in df_price.columns]
    if price_cols:
        df = df.merge(df_price[price_cols], on="fon_kodu", how="left")

    alloc_cols = ["fon_kodu"]
    for col in df_alloc.columns:
        if any(x in str(col).lower() for x in ["hisse", "altın", "devlet", "döviz", "mevduat", "yabancı", "repo", "kıymetli", "tahvil", "bono"]):
            alloc_cols.append(col)
    alloc_cols = list(dict.fromkeys(alloc_cols))
    if len(alloc_cols) > 1:
        df = df.merge(df_alloc[alloc_cols], on="fon_kodu", how="left")

    for col in ["risk", "son_buyukluk", "buyukluk_degisim"] + [c for c in df.columns if c.startswith("getiri_")]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calc_advanced_scores(df, getiri_col="getiri_1y", risk_free=0.40):
    df = df.copy()
    g = df[getiri_col].fillna(0) if getiri_col in df.columns else pd.Series(0, index=df.index)
    r = df["risk"].fillna(4).clip(lower=0.5) if "risk" in df.columns else pd.Series(4, index=df.index)

    df["sharpe"] = (g - risk_free * 0.15) / r
    downside = np.where(g < 0, np.abs(g), 0.01)
    df["sortino"] = (g - risk_free * 0.10) / (downside * r)
    df["calmar"] = g / r

    periods = [c for c in ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y"] if c in df.columns]
    if len(periods) >= 2:
        ranks = df[periods].rank(pct=True)
        df["tutarlilik"] = ranks.mean(axis=1) - ranks.std(axis=1)
    else:
        df["tutarlilik"] = 0.5

    df["skor"] = (
        0.35 * df["sharpe"].rank(pct=True) +
        0.25 * df["sortino"].rank(pct=True) +
        0.20 * df["calmar"].rank(pct=True) +
        0.20 * df["tutarlilik"].rank(pct=True)
    )
    return df


def filter_liquidity(df, min_buyukluk=50e6, max_kuculme=-0.40):
    df = df.copy()
    if "son_buyukluk" in df.columns:
        df = df[(df["son_buyukluk"].isna()) | (df["son_buyukluk"] >= min_buyukluk)]
    if "buyukluk_degisim" in df.columns:
        df = df[(df["buyukluk_degisim"].isna()) | (df["buyukluk_degisim"] >= max_kuculme)]
    return df


def category_similarity(kat1, kat2):
    if pd.isna(kat1) or pd.isna(kat2):
        return 0.3
    k1, k2 = str(kat1).lower(), str(kat2).lower()
    if k1 == k2:
        return 1.0
    hisse_kw = ["hisse", "değişken", "serbest"]
    borc_kw = ["borçlanma", "para piyasası", "katılım"]
    altin_kw = ["altın", "kıymetli", "emtia"]
    yabanci_kw = ["yabancı", "döviz", "eurobond"]
    def group(k):
        if any(x in k for x in hisse_kw): return "hisse"
        if any(x in k for x in borc_kw): return "borc"
        if any(x in k for x in altin_kw): return "altin"
        if any(x in k for x in yabanci_kw): return "yabanci"
        return "diger"
    return 0.6 if group(k1) == group(k2) else 0.15


def estimate_correlation_matrix(funds_df):
    n = len(funds_df)
    corr = np.eye(n)
    kats = funds_df["kategori"].values if "kategori" in funds_df.columns else ["x"] * n
    risks = funds_df["risk"].fillna(4).values if "risk" in funds_df.columns else [4] * n
    for i, j in combinations(range(n), 2):
        sim = category_similarity(kats[i], kats[j])
        risk_diff = abs(risks[i] - risks[j]) / 7
        c = 0.25 + 0.55 * sim - 0.15 * risk_diff
        corr[i, j] = corr[j, i] = np.clip(c, 0.05, 0.95)
    return corr


def portfolio_volatility(weights, risks, corr):
    w = np.array(weights)
    r = np.array(risks)
    vol = r / 7.0
    return np.sqrt(w @ (np.outer(vol, vol) * corr) @ w)


def optimize_weights(returns, risks, corr, method="max_sharpe", risk_free=0.05):
    n = len(returns)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    returns = np.array(returns)
    risks = np.array(risks)

    if method == "equal":
        return np.ones(n) / n
    if method == "risk_parity":
        inv_risk = 1.0 / np.clip(risks, 0.5, 7)
        return inv_risk / inv_risk.sum()
    if method == "min_variance":
        inv_var = 1.0 / np.clip(risks ** 2, 0.25, 49)
        return inv_var / inv_var.sum()

    best_sharpe = -999
    best_w = np.ones(n) / n
    candidates = [np.ones(n) / n]
    inv_r = 1.0 / np.clip(risks, 0.5, 7)
    candidates.append(inv_r / inv_r.sum())
    rng = np.random.default_rng(42)
    for _ in range(80):
        candidates.append(rng.dirichlet(np.ones(n)))

    for w in candidates:
        port_ret = w @ returns
        port_vol = portfolio_volatility(w, risks, corr)
        if port_vol < 1e-6:
            continue
        sharpe = (port_ret - risk_free) / port_vol
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w
    return best_w


def select_diversified(df, n_funds=5, max_per_cat=2):
    if df.empty or "skor" not in df.columns:
        return df.head(0)
    df_sorted = df.sort_values("skor", ascending=False)
    selected, cat_count = [], {}
    for _, row in df_sorted.iterrows():
        kat = row.get("kategori", "Bilinmeyen")
        if pd.isna(kat):
            kat = "Bilinmeyen"
        if cat_count.get(kat, 0) >= max_per_cat:
            continue
        selected.append(row)
        cat_count[kat] = cat_count.get(kat, 0) + 1
        if len(selected) >= n_funds:
            break
    return pd.DataFrame(selected)


def stress_test(weights, categories, returns_1y, scenario="borsa_dususu"):
    w = np.array(weights)
    impact = []
    for kat, ret in zip(categories, returns_1y):
        kat = str(kat).lower() if pd.notna(kat) else ""
        if scenario == "borsa_dususu":
            if any(x in kat for x in ["hisse", "değişken", "serbest"]):
                impact.append(ret * 0.3 - 0.20)
            else:
                impact.append(ret * 0.7 - 0.03)
        elif scenario == "doviz_yukselisi":
            if any(x in kat for x in ["döviz", "yabancı", "eurobond", "serbest"]):
                impact.append(ret * 0.5 + 0.08)
            else:
                impact.append(ret * 0.8 - 0.02)
        elif scenario == "altin_yukselisi":
            if any(x in kat for x in ["altın", "kıymetli", "emtia"]):
                impact.append(ret * 0.4 + 0.12)
            else:
                impact.append(ret * 0.9)
        else:
            impact.append(ret)
    return float(np.sum(w * np.array(impact)))


def calc_portfolio_metrics(df, weights):
    if df.empty or len(weights) == 0:
        return {}
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    m = {}
    for col in ["getiri_1a", "getiri_3a", "getiri_6a", "getiri_1y", "getiri_ybb", "skor", "sharpe", "sortino"]:
        if col in df.columns:
            m[col] = np.sum(df[col].fillna(0).values * w)
    if "risk" in df.columns:
        m["ortalama_risk"] = np.average(df["risk"].fillna(4), weights=w)
    return m


def diversification_score(categories, weights):
    w = np.array(weights)
    w = w / w.sum()
    cat_w = {}
    for c, wi in zip(categories, w):
        c = str(c) if pd.notna(c) else "Diger"
        cat_w[c] = cat_w.get(c, 0) + wi
    hhi = sum(v ** 2 for v in cat_w.values())
    return 1 - hhi


# ==================== SIDEBAR ====================
st.sidebar.title("📊 TEFAS Portföy Pro")
st.sidebar.markdown("---")
page = st.sidebar.radio("Menü", [
    "🏠 Ana Sayfa", "📂 Veri Yükle", "💼 Portföyüm",
    "📈 Analiz & Karşılaştırma", "🤖 Öneri Motoru",
    "🧪 Senaryo & Stres", "ℹ️ Yardım"
])
st.sidebar.markdown("---")

if "df" not in st.session_state:
    st.session_state.df = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

# ==================== SAYFALAR ====================

if page == "🏠 Ana Sayfa":
    st.title("📊 TEFAS Portföy Yöneticisi — Pro")
    st.markdown("""
    **İleri düzey özellikler:**
    - Risk-ayarlı skorlar (Sharpe, Sortino, Calmar, Tutarlılık)
    - Portföy optimizasyonu (Max Sharpe, Min Varyans, Risk Paritesi)
    - Korelasyon tahmini ve çeşitlendirme skoru
    - Senaryo / stres testi
    - Akıllı rebalancing önerileri
    """)
    if st.session_state.df is not None:
        st.success(f"✅ {len(st.session_state.df)} fon yüklü | Portföyde {len(st.session_state.portfolio)} fon")
    else:
        st.warning("Önce **Veri Yükle** sekmesinden dosyaları yükle.")

elif page == "📂 Veri Yükle":
    st.title("📂 Veri Yükleme")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("1️⃣ Getiri & Risk", type=["xlsx", "xls"], key="r")
        f2 = st.file_uploader("2️⃣ Büyüklük", type=["xlsx", "xls"], key="s")
    with c2:
        f3 = st.file_uploader("3️⃣ Genel / Fiyat", type=["xlsx", "xls"], key="p")
        f4 = st.file_uploader("4️⃣ Dağılım", type=["xlsx", "xls"], key="a")
    if st.button("🔄 Birleştir ve Yükle", type="primary"):
        if f1 and f2 and f3 and f4:
            with st.spinner("İşleniyor..."):
                try:
                    st.session_state.df = load_and_prepare_data(f1, f2, f3, f4)
                    st.success(f"✅ {len(st.session_state.df)} fon yüklendi")
                    st.dataframe(st.session_state.df.head(8), use_container_width=True)
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("4 dosyayı da yükle.")

elif page == "💼 Portföyüm":
    st.title("💼 Portföyüm")
    if st.session_state.df is None:
        st.warning("Önce veri yükle.")
    else:
        df = st.session_state.df
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            q = st.text_input("Ara", placeholder="Fon kodu veya adı")
            mask = df["fon_kodu"].str.contains(q, case=False, na=False) | df["fon_adi"].str.contains(q, case=False, na=False) if q else slice(None)
            opts = (df.loc[mask, "fon_kodu"] + " - " + df.loc[mask, "fon_adi"]).tolist()
            sel = st.selectbox("Fon seç", opts)
            kod = sel.split(" - ")[0] if sel else None
        with c2:
            w = st.number_input("Ağırlık %", 0.0, 100.0, 10.0, 1.0)
        with c3:
            st.write(""); st.write("")
            if st.button("➕ Ekle", type="primary") and kod:
                st.session_state.portfolio[kod] = w
                st.rerun()

        if st.session_state.portfolio:
            rows = []
            for k, wt in st.session_state.portfolio.items():
                r = df[df["fon_kodu"] == k]
                if not r.empty:
                    r = r.iloc[0]
                    rows.append({
                        "Fon Kodu": k, "Fon Adı": r.get("fon_adi", ""), "Ağırlık %": wt,
                        "Risk": r.get("risk"), "1Y": r.get("getiri_1y"), "Kategori": r.get("kategori")
                    })
            pdf = pd.DataFrame(rows)
            edited = st.data_editor(pdf, use_container_width=True, key="ed")
            if st.button("💾 Kaydet"):
                st.session_state.portfolio = {row["Fon Kodu"]: row["Ağırlık %"] for _, row in edited.iterrows()}
                st.success("Kaydedildi"); st.rerun()
            rm = st.multiselect("Sil", list(st.session_state.portfolio.keys()))
            if st.button("🗑️ Sil") and rm:
                for k in rm: st.session_state.portfolio.pop(k, None)
                st.rerun()

            ws = [st.session_state.portfolio[k] for k in pdf["Fon Kodu"]]
            tmp = df[df["fon_kodu"].isin(pdf["Fon Kodu"])].copy()
            tmp = calc_advanced_scores(tmp)
            mets = calc_portfolio_metrics(tmp, ws)
            div = diversification_score(pdf["Kategori"].tolist(), ws)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("1Y Getiri", f"%{mets.get('getiri_1y', 0)*100:.1f}")
            m2.metric("Ort. Risk", f"{mets.get('ortalama_risk', 0):.1f}")
            m3.metric("Sharpe", f"{mets.get('sharpe', 0):.2f}")
            m4.metric("Skor", f"{mets.get('skor', 0):.2f}")
            m5.metric("Çeşitlendirme", f"{div:.2f}")

            fig = px.pie(pdf, values="Ağırlık %", names="Fon Kodu", title="Dağılım")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Portföy boş.")

elif page == "📈 Analiz & Karşılaştırma":
    st.title("📈 Analiz & Karşılaştırma")
    if st.session_state.df is None:
        st.warning("Önce veri yükle.")
    else:
        df = st.session_state.df.copy()
        st.sidebar.markdown("### Filtreler")
        if "risk" in df.columns:
            rmin, rmax = st.sidebar.slider("Risk", 1, 7, (1, 7))
            df = df[(df["risk"].between(rmin, rmax)) | df["risk"].isna()]
        if "kategori" in df.columns:
            kats = ["Tümü"] + sorted(df["kategori"].dropna().unique())
            sk = st.sidebar.selectbox("Kategori", kats)
            if sk != "Tümü":
                df = df[df["kategori"] == sk]
        donem_map = {"1 Ay": "getiri_1a", "3 Ay": "getiri_3a", "6 Ay": "getiri_6a", "1 Yıl": "getiri_1y", "YBB": "getiri_ybb"}
        dsel = st.sidebar.selectbox("Dönem", list(donem_map.keys()), index=3)
        gcol = donem_map[dsel]
        minb = st.sidebar.number_input("Min Büyüklük (M TL)", 0, 500, 0) * 1e6
        if minb > 0 and "son_buyukluk" in df.columns:
            df = df[(df["son_buyukluk"].isna()) | (df["son_buyukluk"] >= minb)]

        df = calc_advanced_scores(df, gcol)
        df = df.sort_values("skor", ascending=False)

        show = ["fon_kodu", "fon_adi", "kategori", "risk", gcol, "sharpe", "sortino", "calmar", "tutarlilik", "skor"]
        show = [c for c in show if c in df.columns]
        st.dataframe(df[show].head(40).style.format({
            gcol: lambda x: f"%{x*100:.1f}" if pd.notna(x) else "-",
            "sharpe": "{:.2f}", "sortino": "{:.2f}", "calmar": "{:.2f}",
            "tutarlilik": "{:.2f}", "skor": "{:.2f}"
        }), use_container_width=True)

        if "risk" in df.columns and gcol in df.columns:
            fig = px.scatter(df.dropna(subset=["risk", gcol]), x="risk", y=gcol,
                             hover_name="fon_adi", color="kategori", size="skor",
                             title="Risk vs Getiri (boyut = kompozit skor)")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Öneri Motoru":
    st.title("🤖 Öneri Motoru — Pro")
    if st.session_state.df is None:
        st.warning("Önce veri yükle.")
    else:
        df = st.session_state.df.copy()
        tab1, tab2 = st.tabs(["🎯 Optimize Portföy", "🔄 Rebalancing (Sat-Al)"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            risk_pref = c1.select_slider("Risk Toleransı", [1,2,3,4,5,6,7], 4)
            donem = c2.selectbox("Dönem", ["getiri_1a","getiri_3a","getiri_6a","getiri_1y","getiri_ybb"], 3,
                                 format_func=lambda x: {"getiri_1a":"1A","getiri_3a":"3A","getiri_6a":"6A","getiri_1y":"1Y","getiri_ybb":"YBB"}[x])
            n_funds = c3.slider("Fon sayısı", 3, 10, 5)

            with st.expander("⚙️ Optimizasyon & Filtreler"):
                method = st.selectbox("Optimizasyon Yöntemi", [
                    "max_sharpe", "min_variance", "risk_parity", "equal"
                ], format_func=lambda x: {
                    "max_sharpe": "Maksimum Sharpe", "min_variance": "Minimum Varyans",
                    "risk_parity": "Risk Paritesi", "equal": "Eşit Ağırlık"
                }[x])
                min_b = st.number_input("Min Büyüklük (M TL)", 10, 500, 50)
                max_shrink = st.slider("Max Küçülme %", -100, 0, -40)
                max_cat = st.slider("Kategori başı max fon", 1, 3, 2)

            if st.button("🎯 Optimize Portföy Oluştur", type="primary"):
                filt = df.copy()
                if "risk" in filt.columns:
                    filt = filt[(filt["risk"].between(risk_pref-1, risk_pref+1)) | filt["risk"].isna()]
                filt = filter_liquidity(filt, min_b * 1e6, max_shrink / 100)
                filt = calc_advanced_scores(filt, donem)
                selected = select_diversified(filt, n_funds, max_cat)

                if selected.empty:
                    st.warning("Uygun fon bulunamadı.")
                else:
                    rets = selected[donem].fillna(0).values
                    risks = selected["risk"].fillna(4).values
                    corr = estimate_correlation_matrix(selected)
                    weights = optimize_weights(rets, risks, corr, method=method)
                    weights = weights * 100

                    out = selected.copy()
                    out["Ağırlık %"] = np.round(weights, 1)

                    st.success(f"{len(out)} fon | Yöntem: {method}")
                    st.dataframe(out[["fon_kodu", "fon_adi", "kategori", "risk", donem, "sharpe", "skor", "Ağırlık %"]].style.format({
                        donem: lambda x: f"%{x*100:.1f}" if pd.notna(x) else "-",
                        "sharpe": "{:.2f}", "skor": "{:.2f}"
                    }), use_container_width=True)

                    mets = calc_portfolio_metrics(out, weights)
                    div = diversification_score(out["kategori"].tolist(), weights)
                    corr_avg = (corr.sum() - len(corr)) / (len(corr)**2 - len(corr)) if len(corr) > 1 else 0

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Portföy Getiri", f"%{mets.get(donem, 0)*100:.1f}")
                    m2.metric("Ort. Risk", f"{mets.get('ortalama_risk', 0):.1f}")
                    m3.metric("Sharpe", f"{mets.get('sharpe', 0):.2f}")
                    m4.metric("Çeşitlendirme", f"{div:.2f}")
                    m5.metric("Ort. Korelasyon", f"{corr_avg:.2f}")

                    fig = px.pie(out, values="Ağırlık %", names="fon_kodu", title="Optimize Ağırlıklar")
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button("📥 Portföyüme Aktar"):
                        st.session_state.portfolio = {row["fon_kodu"]: row["Ağırlık %"] for _, row in out.iterrows()}
                        st.success("Aktarıldı!"); st.rerun()

        with tab2:
            if not st.session_state.portfolio:
                st.info("Önce portföy oluştur.")
            else:
                d2 = st.selectbox("Dönem", ["getiri_1a","getiri_3a","getiri_6a","getiri_1y"], 3, key="rb")
                if st.button("🔄 Rebalancing Analizi", type="primary"):
                    dff = calc_advanced_scores(df, d2)
                    curr = dff[dff["fon_kodu"].isin(st.session_state.portfolio.keys())]
                    if curr.empty:
                        st.warning("Portföy fonları veride bulunamadı.")
                    else:
                        curr = curr.copy()
                        curr["mevcut_agirlik"] = curr["fon_kodu"].map(st.session_state.portfolio)
                        curr = curr.sort_values("skor")

                        st.subheader("📉 Zayıf Fonlar (Sat düşünülebilir)")
                        st.dataframe(curr[["fon_kodu","fon_adi","kategori","risk","skor","mevcut_agirlik"]].head(3).style.format({"skor":"{:.2f}"}), use_container_width=True)

                        avg_score = curr["skor"].mean()
                        cands = dff[~dff["fon_kodu"].isin(st.session_state.portfolio.keys())]
                        cands = cands[cands["skor"] > avg_score].sort_values("skor", ascending=False).head(5)

                        st.subheader("📈 Güçlü Alternatifler (Al düşünülebilir)")
                        if cands.empty:
                            st.info("Belirgin daha iyi alternatif yok.")
                        else:
                            st.dataframe(cands[["fon_kodu","fon_adi","kategori","risk","sharpe","skor"]].style.format({
                                "sharpe":"{:.2f}", "skor":"{:.2f}"
                            }), use_container_width=True)

elif page == "🧪 Senaryo & Stres":
    st.title("🧪 Senaryo & Stres Testi")
    if st.session_state.df is None or not st.session_state.portfolio:
        st.warning("Önce veri yükle ve portföy oluştur.")
    else:
        df = st.session_state.df
        port_kods = list(st.session_state.portfolio.keys())
        port_df = df[df["fon_kodu"].isin(port_kods)].copy()
        if port_df.empty:
            st.warning("Portföy fonları bulunamadı.")
        else:
            weights = [st.session_state.portfolio[k] for k in port_df["fon_kodu"]]
            weights = np.array(weights) / sum(weights)
            cats = port_df["kategori"].tolist()
            rets = port_df["getiri_1y"].fillna(0).tolist() if "getiri_1y" in port_df.columns else [0]*len(port_df)

            st.markdown("### Mevcut Portföy Üzerinde Senaryolar")
            scenarios = {
                "borsa_dususu": "Borsa sert düşüşü",
                "doviz_yukselisi": "Döviz yükselişi",
                "altin_yukselisi": "Altın yükselişi"
            }
            results = []
            base = float(np.sum(weights * np.array(rets)))
            results.append({"Senaryo": "Mevcut (1Y baz)", "Tahmini Etki": base})

            for key, label in scenarios.items():
                impact = stress_test(weights, cats, rets, scenario=key)
                results.append({"Senaryo": label, "Tahmini Etki": impact})

            res_df = pd.DataFrame(results)
            res_df["Tahmini Etki %"] = res_df["Tahmini Etki"] * 100
            st.dataframe(res_df[["Senaryo", "Tahmini Etki %"]].style.format({"Tahmini Etki %": "{:.1f}%"}), use_container_width=True)

            fig = px.bar(res_df, x="Senaryo", y="Tahmini Etki %", title="Senaryo Etkileri (kaba tahmin)", color="Tahmini Etki %")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Bu hesaplamalar kategori bazlı kabaca tahminlerdir.")

            st.subheader("Fonlar Arası Tahmini Korelasyon")
            if len(port_df) >= 2:
                corr = estimate_correlation_matrix(port_df)
                corr_df = pd.DataFrame(corr, index=port_df["fon_kodu"], columns=port_df["fon_kodu"])
                fig2 = px.imshow(corr_df, text_auto=".2f", title="Korelasyon Matrisi (tahmini)", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig2, use_container_width=True)

elif page == "ℹ️ Yardım":
    st.title("ℹ️ Yardım")
    st.markdown("""
    ### Metrikler
    - **Sharpe**: (Getiri − risksiz) / Risk
    - **Sortino**: Aşağı yönlü riski daha çok cezalandırır
    - **Calmar**: Getiri / Risk
    - **Tutarlılık**: Farklı dönem performanslarının uyumu
    - **Kompozit Skor**: Yukarıdakilerin birleşimi

    ### Optimizasyon
    - **Max Sharpe**: Risk birimi başına en yüksek getiri
    - **Min Varyans**: En düşük riskli kombinasyon
    - **Risk Paritesi**: Her fonun riske katkısı eşitlenmeye çalışılır
    - **Eşit Ağırlık**: Klasik 1/N

    ### Uyarı
    Bu araç yatırım tavsiyesi değildir.
    """)
