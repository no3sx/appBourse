import streamlit as st
from transformers import pipeline
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import os
import json
from datetime import datetime, timedelta
#streamlit run app_V4.py

# --------- Gestion de la liste d'actions enregistrées ---------
############################################################
#   IA SENTIMENT (chargée UNE SEULE FOIS)
############################################################
@st.cache_resource
def load_ai_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_ai = load_ai_model()

TICKER_FILE = "tickers.json"

DEFAULT_TICKERS = [
    ("Airbus", "AIR.PA"),
    ("Hermès", "RMS.PA"),
    ("Dassault Systèmes", "DSY.PA"),
    ("Sopra Steria", "SOP.PA"),
    ("TotalEnergies", "TTE.PA"),
]


def load_saved_tickers():
    """Charge la liste des tickers enregistrés dans un fichier JSON, ou les 5 par défaut."""
    if os.path.exists(TICKER_FILE):
        try:
            with open(TICKER_FILE, "r") as f:
                data = json.load(f)
            # On enlève les doublons tout en gardant l'ordre
            tickers = list(dict.fromkeys(data))
            return tickers
        except Exception:
            # En cas de problème de lecture, on revient aux valeurs par défaut
            return DEFAULT_TICKERS.copy()
    else:
        return DEFAULT_TICKERS.copy()


def save_saved_tickers(tickers):
    """Sauvegarde la liste des tickers dans le fichier JSON."""
    try:
        with open(TICKER_FILE, "w") as f:
            json.dump(tickers, f)
    except Exception:
        # On ignore les erreurs de sauvegarde pour ne pas casser l'appli
        pass


# =======================
#   FONCTIONS INDICATEURS
# =======================

def compute_moving_averages(df, windows=(20, 50, 200)):
    """Ajoute les moyennes mobiles au DataFrame pour les fenêtres données."""
    df = df.copy()
    for w in windows:
        df[f"MM{w}"] = df["Close"].rolling(window=w).mean()
    return df


def compute_macd(close, short=12, long=26, signal=9):
    """Calcule la MACD classique."""
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(close, window=20, num_std=2):
    """Calcule les bandes de Bollinger (supérieure, moyenne, inférieure)."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def compute_rsi(close, period=14):
    """Calcule le RSI (Relative Strength Index)."""
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain, index=close.index)
    loss = pd.Series(loss, index=close.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stochastic(df, k_window=14, d_window=3):
    """Calcule l'oscillateur stochastique (%K et %D)."""
    low_min = df["Low"].rolling(k_window).min()
    high_max = df["High"].rolling(k_window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_window).mean()
    return k, d


def compute_volume_features(df, window=20):
    """Ajoute des features volume : moyenne mobile de volume et ratio volume."""
    df = df.copy()
    df["Vol_Moy"] = df["Volume"].rolling(window).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_Moy"]
    return df


# ============================
#   FONCTIONS DE NOTATION (0-5)
# ============================

def score_trend(df):
    """Score basé sur la tendance (pente des 30 derniers cours)."""
    if len(df) < 30:
        return 2.5
    closes = df["Close"].tail(30)
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes.values, 1)
    slope = coeffs[0]
    if slope > 0:
        return float(min(5, 2.5 + slope * 50 / closes.mean()))
    else:
        return float(max(0, 2.5 + slope * 50 / closes.mean()))


def score_moving_averages(df):
    """Score basé sur la configuration des MM20 et MM50."""
    last = df.iloc[-1]
    score = 2.5
    if not pd.isna(last.get("MM20")) and not pd.isna(last.get("MM50")):
        if last["Close"] > last["MM20"] > last["MM50"]:
            score = 4.5
        elif last["Close"] < last["MM20"] < last["MM50"]:
            score = 1.5
    return float(score)


def score_macd(macd_line, signal_line):
    """Score simple pour la MACD (position MACD vs signal)."""
    if macd_line is None or signal_line is None:
        return 2.5
    last_macd = macd_line.iloc[-1]
    last_signal = signal_line.iloc[-1]
    if last_macd > last_signal and last_macd > 0:
        return 4.0
    elif last_macd < last_signal and last_macd < 0:
        return 1.5
    else:
        return 2.5


def score_bollinger(close, middle, upper, lower):
    """
    Score dynamique Bollinger (0 à 5)
    - Bas basé sur position du prix dans le canal
    - plus sensible aux excès de volatilité
    """

    if upper is None or lower is None:
        return 2.5

    last_price = close.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    last_middle = middle.iloc[-1]

    if pd.isna(last_upper) or pd.isna(last_lower) or pd.isna(last_middle):
        return 2.5

    # Position relative du prix dans le canal
    position = (last_price - last_lower) / (last_upper - last_lower)

    # Cas extrêmes (breakout)
    if position < -0.2:
        return 4.5   # forte survente → potentiel rebond
    if position > 1.2:
        return 1.0   # surachat massif → risque repli

    # Cas normaux pondérés
    elif position < 0.1:
        return 4.0
    elif position < 0.3:
        return 3.5
    elif position < 0.45:
        return 3.0
    elif position < 0.55:
        return 2.5
    elif position < 0.7:
        return 2.0
    elif position < 0.9:
        return 1.5
    else:
        return 1.2


def score_rsi(rsi_series):
    """Score basé sur le RSI."""
    last_rsi = rsi_series.iloc[-1]
    if pd.isna(last_rsi):
        return 2.5
    if 45 <= last_rsi <= 60:
        return 4.0
    elif last_rsi < 30:
        return 3.0  # survendu, possible rebond
    elif last_rsi > 70:
        return 2.0  # suracheté
    else:
        return 2.5


def score_stochastic(k, d):
    """Score simple pour l'oscillateur stochastique."""
    last_k = k.iloc[-1]
    last_d = d.iloc[-1]
    if pd.isna(last_k) or pd.isna(last_d):
        return 2.5
    if last_k < 20 and last_d < 20:
        return 3.0  # survendu
    elif last_k > 80 and last_d > 80:
        return 2.0  # suracheté
    else:
        return 2.5


def score_volume(df):
    """Score basé sur l'activité de volume."""
    last = df.iloc[-1]
    if pd.isna(last.get("Vol_Ratio")):
        return 2.5
    r = last["Vol_Ratio"]
    if r > 1.5:
        return 4.0  # gros volume → intérêt
    elif r < 0.7:
        return 2.0  # peu de volume → intérêt faible
    else:
        return 2.5


def score_candles():
    """Score neutre pour les chandeliers (placeholder)."""
    return 2.5


def score_head_shoulders():
    """Score neutre pour la figure épaule-tête-épaule (placeholder)."""
    return 2.5


def score_news_placeholder():
    """Score neutre pour les actualités (en attendant une vraie analyse NLP)."""
    return 2.5


############################################################
#   NEWS + IA + IMPORTANCE
############################################################


# Mapping ticker → vrai nom entreprise
TICKER_TO_NAME = {
    "AIR.PA": "Airbus",
    "RMS.PA": "Hermes International",
    "DSY.PA": "Dassault Systemes SE",
    "TTE.PA": "TotalEnergies SE",
    "SOP.PA": "Sopra",
}



import requests

NEWSAPI_KEY = "cd5cc157ee04417da78716753d177345"

def get_news(ticker):
    """
    Actualités via NewsAPI.org (API très stable et fiable).
    Structure identique à MediaStack → aucune modification dans le reste de l'app.
    """

    company = TICKER_TO_NAME.get(ticker, ticker)

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": company,
        "language": "fr",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWSAPI_KEY,
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "articles" not in data or len(data["articles"]) == 0:
        return []

    news_list = []

    for article in data["articles"][:5]:

        title = article.get("title", "")
        link = article.get("url", "")
        source = article.get("source", {}).get("name", "Inconnu")
        date = article.get("publishedAt", "")[:10]

        # Sentiment IA identique à ton app
        pred = sentiment_ai(title)[0]
        lab = pred["label"].upper()
        score = float(pred["score"])

        if "POS" in lab:
            sentiment = "Positive"
            sentiment_score = 3 + 2 * score
        elif "NEG" in lab:
            sentiment = "Negative"
            sentiment_score = 2 - 2 * score
        else:
            sentiment = "Neutral"
            sentiment_score = 2.5

        sentiment_score = max(0, min(5, sentiment_score))

        importance = 1
        final_score = sentiment_score

        news_list.append({
            "title": title,
            "link": link,
            "publisher": source,
            "date": date,
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "importance": importance,
            "final_score": round(final_score, 2),
            "sentiment_comment": lab
        })

    return news_list



# =====================
#   FONCTION DONNÉES
# =====================

@st.cache_data
def load_data(ticker, period, interval):
    """Télécharge les données OHLCV depuis Yahoo Finance, en nettoyant les colonnes."""
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if data.empty:
        raise ValueError("Aucune donnée reçue. Vérifie le ticker, la période ou ta connexion.")

    # Si colonnes en MultiIndex, on ne garde que le niveau le plus bas
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Certains cas n'ont que 'Adj Close' -> on renomme en 'Close'
    if "Close" not in data.columns and "Adj Close" in data.columns:
        data = data.rename(columns={"Adj Close": "Close"})

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}. "
            f"Colonnes disponibles : {list(data.columns)}"
        )

    return data[required].copy()


# =====================
#   INTERFACE STREAMLIT
# =====================

st.set_page_config(page_title="Modèle d'analyse technique", layout="wide")

st.title("📈 Modèle d'analyse technique – Projet Finance")

#st.markdown(
#    """
#    Interface web pour :
#    - choisir une action  
#    - analyser la courbe  
#    - appliquer différentes méthodes d'analyse technique  
#    - obtenir une **note par méthode**, une **note actualité** et une **note globale**.
#    """
#)

# ----- Gestion de l'état de l'analyse -----
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --------- Barre latérale (contrôles) ---------

saved_tickers = load_saved_tickers()

st.sidebar.header("⚙️ Paramètres")

# 1) Choix dans la liste existante
# Liste affichée = noms humains
# Liste des noms visibles dans le menu
company_names = [name for name, ticker in DEFAULT_TICKERS]

# Choix utilisateur
selected_company = st.sidebar.selectbox("Actions enregistrées", company_names)

# Convertir nom → ticker
ticker = dict(DEFAULT_TICKERS)[selected_company]



# 2) OU saisie d'un nouveau ticker
new_ticker_input = st.sidebar.text_input(
    "Ou saisir un nouveau ticker (Yahoo Finance)",
    value="",
    placeholder="Ex : MC.PA, OR.PA, TSLA, NVDA, ..."
)

# Ticker final utilisé
if new_ticker_input.strip():
    ticker = new_ticker_input.strip().upper()
else:
    pass

# Choix de la période / intervalle
period = st.sidebar.selectbox(
    "Période",
    options=["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

interval = st.sidebar.selectbox(
    "Intervalle",
    options=["1m", "5m", "15m", "30m", "1h", "1d", "1wk"],
    index=5
)

# Choix des méthodes d'analyse technique
methods = st.sidebar.multiselect(
    "Méthodes d'analyse technique",
    options=[
        "Tendances",
        "Moyennes Mobiles",
        "MACD",
        "Bollingers",
        "Stochastique",
        "Chandeliers",
        "RSI",
        "Volumes (partie 1)",
        "Volumes (partie 2)",
        "Épaule-Tête-Épaule",
    ],
    default=["Bollingers"]
)

# Bouton pour lancer l'analyse
if st.sidebar.button("Lancer l'analyse"):
    st.session_state.analysis_done = True

# =========================
#   BLOC PRINCIPAL ANALYSE
# =========================

if st.session_state.analysis_done:
    try:
        # -------- Chargement des données principales --------
        df = load_data(ticker, period, interval)

        # Si l'utilisateur a saisi un nouveau ticker valide, on l'ajoute à la base
        if new_ticker_input.strip():
            upper_ticker = new_ticker_input.strip().upper()
            if upper_ticker not in saved_tickers:
                saved_tickers.append(upper_ticker)
                save_saved_tickers(saved_tickers)
                st.sidebar.success(f"Ticker '{upper_ticker}' ajouté aux actions enregistrées.")

        # -------- Calcul des indicateurs techniques --------
        df = compute_moving_averages(df)
        upper, middle, lower = compute_bollinger(df["Close"])
        macd_line, signal_line, macd_hist = compute_macd(df["Close"])
        rsi_series = compute_rsi(df["Close"])
        k, d = compute_stochastic(df)
        df = compute_volume_features(df)

        # ----------- Affichage courbe principale (PLOTLY) -----------
        st.subheader(f"📊 Courbe de l'action {ticker}")

        # Création de la figure Plotly
        fig = go.Figure()

        # Courbe du cours de clôture
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close"
            )
        )

        # Ajout des moyennes mobiles si la méthode est sélectionnée
        if "Moyennes Mobiles" in methods:
            for w in [20, 50, 200]:
                col = f"MM{w}"
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode="lines",
                            name=col
                        )
                    )

        # Ajout des bandes de Bollinger si sélectionné
        if "Bollingers" in methods:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper,
                    mode="lines",
                    name="Bollinger Haut",
                    line=dict(dash="dash")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower,
                    mode="lines",
                    name="Bollinger Bas",
                    line=dict(dash="dash")
                )
            )

        # Mise en forme générale du graphique
        fig.update_layout(
            title=f"Cours de {ticker}",
            xaxis_title="Date",
            yaxis_title="Prix",
            template="plotly_dark",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------- Données importantes ----------
        st.subheader("📌 Données importantes")

        # 1) Performance sur la période sélectionnée (df = historique choisi, ex: 6mo en 1d)
        first_close = float(df["Close"].iloc[0])
        last_close_period = float(df["Close"].iloc[-1])
        perf_periode = (last_close_period / first_close - 1) * 100

        # 2) Données jour : on prend l'historique des 2 derniers jours en daily
        daily = yf.download(ticker, period="2d", interval="1d")

        if (daily is not None) and (len(daily) >= 2):
            prev_close = float(daily["Close"].iloc[-2])   # clôture veille
            last_price = float(daily["Close"].iloc[-1])   # cours du jour (ou dernière clôture)
            perf_jour = (last_price / prev_close - 1) * 100
        else:
            # Si on n'a pas assez de données, on se rabat sur la période
            last_price = last_close_period
            perf_jour = 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Dernier cours", f"{last_price:.2f} €")
        col2.metric("Perf période", f"{perf_periode:.2f}%")
        col3.metric("Variation jour", f"{perf_jour:.2f}%")

        # ---------- Notes par méthode ----------
        st.subheader("🧮 Notes par méthode d'analyse (0 à 5)")

        notes = {}

        if "Tendances" in methods:
            notes["Tendances"] = score_trend(df)

        if "Moyennes Mobiles" in methods:
            notes["Moyennes Mobiles"] = score_moving_averages(df)

        if "MACD" in methods:
            notes["MACD"] = score_macd(macd_line, signal_line)

        if "Bollingers" in methods:
            notes["Bollingers"] = score_bollinger(df["Close"], middle, upper, lower)

        if "RSI" in methods:
            notes["RSI"] = score_rsi(rsi_series)

        if "Stochastique" in methods:
            notes["Stochastique"] = score_stochastic(k, d)

        if "Volumes (partie 1)" in methods or "Volumes (partie 2)" in methods:
            notes["Volumes"] = score_volume(df)

        if "Chandeliers" in methods:
            notes["Chandeliers"] = score_candles()

        if "Épaule-Tête-Épaule" in methods:
            notes["Épaule-Tête-Épaule"] = score_head_shoulders()

        if notes:
            notes_df = pd.DataFrame(
                [{"Méthode": k, "Note": round(v, 2)} for k, v in notes.items()]
            )
            st.table(notes_df)

        # ---------- Note actualité ----------
        ############################################################
        #   NEWS IA + NOTE
        ############################################################

        st.subheader("📰 Actualités – Analyse IA & Importance")

        news = get_news(ticker)
        total_news_score = 0

        if len(news) == 0:
            st.info("Aucune actualité disponible pour ce ticker.")
            news_score = 2.5
        else:
            for n in news:
                # ======== NORMALISATION DU SENTIMENT (0–5) ========
                label = n.get("sentiment", "Neutral")
                raw_sent = float(n.get("sentiment_score", 2.5))

                # Corrige les dérives éventuelles venant du modèle GPT
                if label == "Positive":
                    sentiment_score = min(5, 4 + 1 * (raw_sent / 5))
                elif label == "Negative":
                    sentiment_score = max(0, 1 - 1 * (raw_sent / 5))
                else:
                    sentiment_score = 2.5

                # ======== IMPORTANCE (1 à 3 → bonus limité) ========
                importance = int(n.get("importance", 1))
                bonus = {1: 0, 2: 0.5, 3: 1}.get(importance, 0)

                # ======== SCORE FINAL (0–5) ========
                final_score = max(0, min(sentiment_score + bonus, 5))

                # ======== AFFICHAGE ========
                # ======== AFFICHAGE ========

                # 🔗 Titre cliquable + lien corrigé
                link = n.get('link', '')
                if link.startswith("//"):
                    link = "https:" + link
                elif link.startswith("www."):
                    link = "https://" + link
                if not link.startswith("http"):
                    link = "https://" + link

                st.markdown(f"### 📎 [{n['title']}]({link})")

                # 📰 Source + date
                date = n.get("date", "—")
                st.write(f"🔗 **Source :** {n['publisher']}")
                st.write(f"📅 **Date :** {date}")

                # 🤖 Sentiment IA + Score
                st.write(f"🧠 Sentiment IA : **{label}** – Score IA corrigé : **{sentiment_score:.2f} / 5**")

                # 📊 Importance + score final
                st.write(f"📊 Importance : **{importance}** – Score final : **{final_score:.2f}**")

                st.markdown("---")

                total_news_score += final_score


            news_score = total_news_score / len(news)

            st.metric("Note actualités (0–5)", f"{news_score:.2f}")

        # ---------- Note globale ----------
        if notes:
            moyenne_technique = float(np.mean(list(notes.values())))
        else:
            moyenne_technique = 2.5

        note_globale = 0.8 * moyenne_technique + 0.2 * news_score

        st.subheader("🎯 Note globale")
        st.metric("Globale (0–5)", f"{note_globale:.2f}")

        # Seuils 1.5 / 3.5
        if note_globale < 1.5:
            decision = "Vendre"
        elif note_globale < 3.5:
            decision = "Ne rien faire"
        else:
            decision = "Acheter"

        st.subheader("🧭 Recommandation finale")
        st.metric("Décision", decision)


    ############################################################

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Choisis un ticker, une période, puis clique sur **Lancer l'analyse**.")
