"""Feature engineering pré-match pour la prédiction de résultats LaLiga.

Une seule fonction publique, ``build_pre_match_features``, transforme les
matchs bruts fusionnés (un match par ligne) en un tableau de features
disponibles AVANT le coup d'envoi, plus la cible ``target``.

Point clé anti-fuite : pour chaque match, les features (Elo, forme, repos)
sont lues AVANT la mise à jour de l'Elo et de l'historique. Un match n'utilise
donc jamais d'information issue de lui-même ni du futur.
"""

import pandas as pd

# Correspondance jour de la semaine -> entier. Constante du module : la
# fonction ne dépend plus d'aucune variable globale définie ailleurs.
DAY_TO_INT = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}


def build_pre_match_features(df_all, k_elo=20, n_form=5, base_elo=1500):
    """Construit les features pré-match à partir des matchs fusionnés.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Matchs bruts fusionnés. Colonnes attendues : ``Date``, ``Day``,
        ``home_team``, ``away_team``, ``xG_home``, ``xG_away``,
        ``score_home``, ``score_away``.
    k_elo : int, default 20
        Facteur K du système Elo (amplitude de la mise à jour par match).
    n_form : int, default 5
        Nombre de matchs récents utilisés pour les features de forme.
    base_elo : float, default 1500
        Elo initial attribué à chaque équipe.

    Returns
    -------
    pandas.DataFrame
        Une ligne par match, avec les colonnes de features pré-match et la
        cible ``target`` (0 = défaite à domicile, 1 = nul, 2 = victoire à
        domicile).
    """
    # Copie défensive + tri chronologique : la fonction est ainsi robuste,
    # que l'appelant ait déjà trié / converti la colonne Date ou non.
    df = df_all.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    elo = {t: base_elo for t in teams}
    history = {t: [] for t in teams}
    last_date = {t: None for t in teams}
    rows = []

    def form(team, n=n_form):
        """Moyenne (points, goal diff, xG) sur les n derniers matchs joués."""
        h = history[team][-n:]
        if len(h) == 0:
            return 0.0, 0.0, 0.0
        pts = sum(x["pts"] for x in h) / len(h)
        gd = sum(x["gd"] for x in h) / len(h)
        xg = sum(x["xG"] for x in h) / len(h)
        return float(pts), float(gd), float(xg)

    for i in range(len(df)):
        row = df.iloc[i]

        date = row["Date"]
        day = DAY_TO_INT[row["Day"]]
        home, away = row["home_team"], row["away_team"]
        xG_home, xG_away = row["xG_home"], row["xG_away"]
        gh, ga = row["score_home"], row["score_away"]

        # --- Features pré-match (figées AVANT toute mise à jour) ---
        elo_h, elo_a = elo[home], elo[away]
        ph, gdh, xGh = form(home)
        pa, gda, xGa = form(away)
        rh = (date - last_date[home]).days if last_date[home] is not None else 7
        ra = (date - last_date[away]).days if last_date[away] is not None else 7

        # --- Label + scores Elo réalisés ---
        if gh > ga:
            y, Sh, Sa = 2, 1.0, 0.0
        elif gh == ga:
            y, Sh, Sa = 1, 0.5, 0.5
        else:
            y, Sh, Sa = 0, 0.0, 1.0

        rows.append({
            "date": date,
            "day": day,
            "home_team": home,
            "away_team": away,
            "elo_home": elo_h,
            "elo_away": elo_a,
            "elo_diff": elo_h - elo_a,
            "form_xG_home_5": xGh,
            "form_xG_away_5": xGa,
            "xG_diff": xGh - xGa,
            "form_points_home_5": ph,
            "form_points_away_5": pa,
            "form_goal_diff_home_5": gdh,
            "form_goal_diff_away_5": gda,
            "rest_diff": rh - ra,
            "target": y,
        })

        # --- Mise à jour Elo + historique (APRÈS avoir figé les features) ---
        expected_home = 1 / (1 + 10 ** (-(elo_h - elo_a) / 400))
        elo[home] += k_elo * (Sh - expected_home)
        elo[away] += k_elo * (Sa - (1 - expected_home))
        history[home].append({"pts": Sh * 3, "gd": gh - ga, "xG": xG_home})
        history[away].append({"pts": Sa * 3, "gd": ga - gh, "xG": xG_away})
        last_date[home] = date
        last_date[away] = date

    return pd.DataFrame(rows)
