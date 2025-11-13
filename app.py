import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f, t, norm

# --- Zaokrkouhlování ---
def fmt(x):
    if abs(x) < 1e-6:
        return "0.0000"
    else:
        return f"{x:.4g}"

# --- Sidebar: hlavní volba aplikace ---
st.sidebar.title("Navigace")
main_page = st.sidebar.radio("Vyber aplikaci", ["Úkoly 1–6", "Úkoly 7–11"])

# =========================================================
# BLOK 1: Úkoly 1–6
# =========================================================
if main_page == "Úkoly 1–6":
    st.title("Ekonometrie - Úkol 1 - první část")
    page = st.sidebar.radio("Vyber úkol", [
        "Úvod a data", "Model OLS", "Úloha 1", "Úloha 2", "Úloha 3",
        "Úloha 4", "Úloha 5", "Úloha 6"
    ])
    uploaded_file = st.file_uploader("Nahraj CSV soubor pro Úkoly 1–6", type="csv", key="file_app1")
    if uploaded_file is not None:
        st.session_state["data_app1"] = pd.read_csv(uploaded_file)
    if "data_app1" in st.session_state:
        data = st.session_state["data_app1"]
        data["log.price"] = np.log(data["price"])
        data["log.lotsize"] = np.log(data["lotsize"])
        X = data[["log.lotsize", "bedrooms", "driveway", "fullbase", "airco"]]
        X = sm.add_constant(X)
        y = data["log.price"]
        model = sm.OLS(y, X).fit()

        if page == "Úvod a data":
            st.header("Úvod a data")
            st.dataframe(data)  

        elif page == "Model OLS":
            st.header("Model OLS")
            st.subheader("📄 Výsledky regresního modelu (OLS summary)")
            st.code(model.summary(), language="text")

        elif page == "Úloha 1":
            st.header("Úloha 1")
            st.metric("R²", f"{fmt(model.rsquared * 100)} %")
            st.metric("Upravené R²", f"{fmt(model.rsquared_adj * 100)} %")

        elif page == "Úloha 2":
            st.header("Úloha 2")
            coef = model.params["log.lotsize"]
            conf_int = model.conf_int().loc["log.lotsize"]
            st.metric("Bodový odhad", f"{fmt(coef)} %")
            st.metric("95% interval spolehlivosti", f"{fmt(conf_int[0])} % - {fmt(conf_int[1])} %")

        elif page == "Úloha 3":
            st.header("Úloha 3")
            st.markdown("**Hypotézy:**  \nH0: β_driveway = 0  \nH1: β_driveway > 0")
            coef_driveway = model.params["driveway"]
            p_two_sided = model.pvalues["driveway"]
            p_one_sided = p_two_sided / 2 if coef_driveway > 0 else 1 - p_two_sided / 2
            st.metric("Jednostranná p-hodnota", f"{fmt(p_one_sided)}")
            decision = "Zamítáme H0" if p_one_sided < 0.05 else "Nezamítáme H0"
            st.metric("Rozhodnutí (α=5%)", decision)

        elif page == "Úloha 4":
            st.header("Úloha 4")
            R = np.zeros(len(model.params))
            R[list(model.params.index).index("log.lotsize")] = 1
            q = 1
            f_test = model.f_test((R, q))
            st.markdown("**Hypotézy:**  \nH0: β_log(lotsize) = 1  \nH1: β_log(lotsize) ≠ 1")
            st.metric("F-statistika", f"{fmt(float(f_test.fvalue))}")
            st.metric("p-hodnota", f"{fmt(float(f_test.pvalue))}")
            decision = "Zamítáme H0" if float(f_test.pvalue) < 0.01 else "Nezamítáme H0"
            st.metric("Rozhodnutí (α=1%)", decision)

        elif page == "Úloha 5":
            st.header("Úloha 5")
            dummy_vars = ["driveway", "fullbase", "airco"]
            R = np.zeros((len(dummy_vars), len(model.params)))
            for i, var in enumerate(dummy_vars):
                R[i, list(model.params.index).index(var)] = 1
            q = np.zeros(len(dummy_vars))
            f_test = model.f_test((R, q))
            F_stat = float(f_test.fvalue)
            dfn = len(dummy_vars)
            dfd = int(model.df_resid)
            alpha = 0.01
            F_crit = f.ppf(1 - alpha, dfn, dfd)
            st.metric("(a) Hodnota testové statistiky", f"{fmt(F_stat)}")
            st.metric("(b) Stupně volnosti čitatele", f"{dfn}")
            st.metric("(c) Stupně volnosti jmenovatele", f"{dfd}")
            st.metric(f"(d) Kritická hodnota (α={alpha*100:.0f}%)", f"{fmt(F_crit)}")
            decision = "Zamítáme H0" if float(f_test.pvalue) < 0.01 else "Nezamítáme H0"
            st.metric("Rozhodnutí (α=1%)", decision)

        elif page == "Úloha 6":
            st.header("Úloha 6")
            R = np.zeros((1, len(model.params)))
            R[0, list(model.params.index).index("fullbase")] = 1
            R[0, list(model.params.index).index("airco")] = -1
            q = np.zeros(1)
            f_test = model.f_test((R, q))
            st.markdown("**Hypotézy:**  \nH0: β_fullbase = β_airco  \nH1: β_fullbase ≠ β_airco")
            st.metric("p-hodnota", f"{fmt(float(f_test.pvalue))}")
            decision = "Zamítáme H0" if float(f_test.pvalue) < 0.05 else "Nezamítáme H0"
            st.metric("Rozhodnutí (α=5%)", decision)

# =========================================================
# BLOK 2: Úkoly 7–11
# =========================================================
elif main_page == "Úkoly 7–11":
    st.title("Ekonometrie - Úkol 1 - druhá část")
    page = st.sidebar.radio("Vyber úkol", [
        "Úvod a data", "Model OLS", "Úloha 7", "Úloha 8", "Úloha 9",
        "Úloha 10", "Úloha 11"
    ])
    uploaded_file = st.file_uploader("Nahraj CSV soubor pro Úkoly 7–11", type="csv", key="file_app2")
    if uploaded_file is not None:
        st.session_state["data_app2"] = pd.read_csv(uploaded_file)
    if "data_app2" in st.session_state:
        data = st.session_state["data_app2"]
        data["l_wage"] = np.log(data["wage"])
        data["sq_age"] = data["age"] ** 2
        data["female_urban"] = data["female"] * data["urban"]
        y = data["l_wage"]
        X = data[["age", "sq_age", "educ", "married", "female", "urban", "female_urban"]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        if page == "Úvod a data":
            st.header("Úvod a data")
            st.dataframe(data)

        elif page == "Model OLS":
            st.header("Model OLS")
            st.code(model.summary(), language="text")

        elif page == "Úloha 7":
            st.header("Úloha 7")
            beta1 = model.params["age"]
            beta2 = model.params["sq_age"]
            turning_point = -beta1 / (2 * beta2)
            shape = "tvar U" if beta2 > 0 else "obrácený tvar U"
            st.metric("Bod zlomu (věk)", fmt(turning_point))
            st.metric("Tvar křivky", shape)

        elif page == "Úloha 8":
            st.header("Úloha 8")
            beta1 = model.params["age"]
            beta2 = model.params["sq_age"]
            def pct_effect_at(age):
                return 100 * (beta1 + 2 * beta2 * age)
            for a in [35, 45]:
                st.metric(f"Věk {a}", f"{fmt(pct_effect_at(a))} %")

        elif page == "Úloha 9":
            st.header("Úloha 9")
            b_f = model.params["female"]
            b_fu = model.params["female_urban"]
            cov = model.cov_params()
            var_f = cov.loc["female", "female"]
            var_fu = cov.loc["female_urban", "female_urban"]
            cov_f_fu = cov.loc["female", "female_urban"]
            z = norm.ppf(0.975)
            theta_rural = b_f
            se_rural = np.sqrt(var_f)
            pct_rural = (np.exp(theta_rural) - 1) * 100
            ci_rural_low = (np.exp(theta_rural - z * se_rural) - 1) * 100
            ci_rural_up = (np.exp(theta_rural + z * se_rural) - 1) * 100
            theta_urban = b_f + b_fu
            se_urban = np.sqrt(var_f + var_fu + 2 * cov_f_fu)
            pct_urban = (np.exp(theta_urban) - 1) * 100
            ci_urban_low = (np.exp(theta_urban - z * se_urban) - 1) * 100
            ci_urban_up = (np.exp(theta_urban + z * se_urban) - 1) * 100
            st.metric("Vesnice", f"{fmt(pct_rural)} % [{fmt(ci_rural_low)} %; {fmt(ci_rural_up)} %]")
            st.metric("Město", f"{fmt(pct_urban)} % [{fmt(ci_urban_low)} %; {fmt(ci_urban_up)} %]")

        elif page == "Úloha 10":
            st.header("Úloha 10")
            b_fu = model.params["female_urban"]
            se_fu = model.bse["female_urban"]
            t_stat = b_fu / se_fu
            df_resid = model.df_resid
            p_value = 2 * (1 - t.cdf(abs(t_stat), df_resid))
            st.markdown("**Hypotézy:**  \nH0: β_(female × urban) = 0  \nH1: β_(female × urban) ≠ 0")
            st.metric("p-hodnota", f"{fmt(p_value)} %")
            decision = "Zamítáme H0" if p_value < 0.05 else "Nezamítáme H0"
            st.metric("Rozhodnutí (α=5%)", decision)

        elif page == "Úloha 11":
            st.header("Úloha 11")
            new_obs = pd.DataFrame({
                "const": [1],
                "age": [30],
                "sq_age": [30**2],
                "educ": [13],
                "married": [1],
                "female": [0],
                "urban": [1],
                "female_urban": [0]
            })
            log_pred = model.predict(new_obs)[0]
            sigma2 = model.mse_resid
            wage_normality = np.exp(log_pred + sigma2 / 2)
            smearing_factor = np.mean(np.exp(model.resid))
            wage_duan = np.exp(log_pred) * smearing_factor
            st.metric("Mzda (normalita)", f"{fmt(wage_normality)}")
            st.metric("Mzda (Duan)", f"{fmt(wage_duan)}")