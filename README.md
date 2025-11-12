# Streamlit aplikace pro ekonometrické úlohy

## Popis
Tato aplikace je vytvořena ve **Streamlit** a slouží k řešení úloh z ekonometrie, zejména:
- OLS regresní modely
- Testy hypotéz
- Predikce

Aplikace obsahuje dvě hlavní části:
- **Úkoly 1–6:** Analýza cen nemovitostí
- **Úkoly 7–11:** Analýza mezd

---

## Instalace
1. Naklonujte repozitář:
   ```bash
   git clone <URL vašeho repozitáře>
   cd <název složky>
   ```
2. Nainstalujte závislosti:
   ```bash
   pip install -r requirements.txt
   ```

---

## Spuštění aplikace
Spusťte aplikaci pomocí příkazu:
```bash
streamlit run app.py
```

---

## Struktura projektu
```
├── app.py              # Hlavní soubor aplikace
├── requirements.txt    # Seznam závislostí
├── data/               # Datové soubory pro analýzu
└── README.md           # Dokumentace projektu
```

---

## Požadavky
- Python 3.12.10
- Streamlit, Pandas, NumPy, Statsmodels, Matplotlib (viz `requirements.txt`)

---

## Autor
Tomáš Smažík
