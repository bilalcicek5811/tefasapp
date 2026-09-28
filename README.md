# TEFAS Portföy Yöneticisi

TEFAS fon verilerini analiz eden, portföy oluşturup öneri sunan Streamlit uygulaması.

## Özellikler

- 4 farklı TEFAS Excel dosyasını yükleyip birleştirme
- Portföy oluşturma ve yönetme (ağırlık ayarlama)
- Risk ve getiri bazlı filtreleme / sıralama
- Risk-Getiri matrisi
- Kategori bazlı performans
- Risk toleransına göre otomatik portföy önerisi

## Kurulum (Yerel)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud'a Deploy

1. Bu repoyu GitHub'a yükle
2. [share.streamlit.io](https://share.streamlit.io) adresine git
3. GitHub hesabınla giriş yap
4. Bu repoyu seç → `app.py` dosyasını işaretle → Deploy

## Gerekli Dosyalar (TEFAS'tan)

1. Getiri & Risk dosyası (Menkul Kıymet Yatırım Fonları)
2. Büyüklük dosyası
3. Genel (Fiyat) dosyası
4. Dağılım dosyası

## Not

Bu uygulama yatırım tavsiyesi değildir.
