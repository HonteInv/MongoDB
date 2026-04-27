"""
stress_scenarios.py — Curated historical stress scenario library.

PURPOSE
-------
Provides a pre-defined set of historical stress scenarios relevant to a
global macro portfolio with the following dominant risk factors:
    - Long duration / rates (DV01 ~$464k as of Feb 2026)
    - Long USDCNH (~$167M notional)
    - Short CHFJPY (~-$158M notional)
    - Long platinum / commodities (~$47M notional)
    - Long gold miners / equities
    - Long crypto (IBIT)
    - Short SPX hedge

HOW IT WILL BE USED (pending supervisor review)
------------------------------------------------
1. RiskAnalystAgent._compute_risk_metrics() identifies the dominant themes
   (e.g. rates = 100% of P&L, FX = -62%)
2. select_scenarios(dominant_themes) picks the 2 most relevant scenarios
   from the library below
3. Those 2 scenarios are injected into the risk agent prompt as structured
   context — Claude applies them to the actual position data
4. Claude writes a 3rd scenario itself (forward-looking hypothetical)

This prevents Claude from free-picking scenarios (e.g. always citing 2008
GFC even when the portfolio has no credit exposure).

NOT YET WIRED IN — confirm with supervisor before integrating.

FIELD DEFINITIONS
-----------------
name          : Display name shown in the report
period        : Calendar dates when the scenario occurred
context       : 2-3 sentence description of what caused it
market_moves  : Approximate peak-to-trough moves during the event
    us_10y_bps      : US 10-year Treasury yield change (basis points, +/-)
    us_2y_bps       : US 2-year yield change (basis points) — front end sensitivity
    usd_dxy_pct     : US Dollar Index change (percent)
    usdcnh_pct      : USD/CNH spot change (percent, + = USD stronger / CNH weaker)
    usdjpy_pct      : USD/JPY change (percent, + = USD stronger / JPY weaker)
    chfjpy_pct      : CHF/JPY change (percent, + = CHF stronger vs JPY)
    equities_pct    : Broad equity index change (percent, S&P 500 unless noted)
    gold_pct        : Gold spot price change (percent)
    platinum_pct    : Platinum spot price change (percent)
    commodities_pct : Broad commodity index change (percent, CRB or similar)
    crypto_pct      : Bitcoin/broad crypto change (percent)
    vol_vix         : Peak VIX level reached during scenario
relevant_for  : List of portfolio themes this scenario primarily stresses
    options: "rates", "FX", "equity", "commodity", "crypto", "vol"
"""

# ============================================================
# Scenario Library
# ============================================================

STRESS_SCENARIOS: dict[str, dict] = {

    # ----------------------------------------------------------
    # 1. 2022 US Rate Shock
    # Most relevant for: long duration (DV01), USDCNH, gold/platinum
    # ----------------------------------------------------------
    "2022_rate_shock": {
        "name": "2022 US Rate Shock",
        "period": "January – October 2022",
        "context": (
            "The Fed pivoted sharply from quantitative easing to the most aggressive "
            "hiking cycle in 40 years, delivering 425bps of hikes. CPI peaked at 9.1% "
            "in June. The simultaneous selloff in both bonds and equities eliminated "
            "traditional 60/40 diversification benefits."
        ),
        "market_moves": {
            "us_10y_bps":       +280,   # 1.50% → 4.30%
            "us_2y_bps":        +370,   # front end moved more (curve inversion)
            "usd_dxy_pct":      +15,    # DXY from ~96 to ~114
            "usdcnh_pct":       +12,    # CNH weakened as Fed/PBOC diverged
            "usdjpy_pct":       +30,    # JPY collapsed on BOJ yield cap
            "chfjpy_pct":       -5,     # CHF also weakened but less than JPY
            "equities_pct":     -25,    # S&P 500 peak to trough
            "gold_pct":         -10,    # gold underperformed despite inflation
            "platinum_pct":     -18,    # industrial demand fears
            "commodities_pct":  +30,    # energy surge offset metals weakness
            "crypto_pct":       -65,    # crypto entered prolonged bear market
            "vol_vix":          35,
        },
        "relevant_for": ["rates", "FX", "commodity", "crypto", "equity"],
    },

    # ----------------------------------------------------------
    # 2. 2015 CNY Devaluation
    # Most relevant for: USDCNH long, EM FX, commodities
    # ----------------------------------------------------------
    "2015_cny_devaluation": {
        "name": "2015 CNY Devaluation & EM Stress",
        "period": "August – December 2015",
        "context": (
            "China's surprise devaluation of the yuan in August 2015 triggered a global "
            "risk-off episode. Capital outflows from China accelerated, commodity prices "
            "collapsed on China demand fears, and EM currencies broadly weakened. "
            "The PBOC burned through ~$500bn of reserves defending the currency."
        ),
        "market_moves": {
            "us_10y_bps":       -20,    # mild flight to quality
            "us_2y_bps":        -10,
            "usd_dxy_pct":      +5,
            "usdcnh_pct":       +4,     # initial devaluation; then stabilised
            "usdjpy_pct":       -3,     # JPY strengthened on risk-off
            "chfjpy_pct":       +5,     # CHF also safe-haven bid
            "equities_pct":     -12,    # S&P 500 correction (Aug flash crash)
            "gold_pct":         -5,
            "platinum_pct":     -20,    # platinum hit hard on China demand fears
            "commodities_pct":  -20,    # CRB index fell sharply
            "crypto_pct":       None,   # not relevant in 2015
            "vol_vix":          40,     # VIX spiked to 40 in August
        },
        "relevant_for": ["FX", "commodity", "equity"],
    },

    # ----------------------------------------------------------
    # 3. 2018 US–China Trade War
    # Most relevant for: USDCNH, EM FX, equities, platinum (auto demand)
    # ----------------------------------------------------------
    "2018_trade_war": {
        "name": "2018 US–China Trade War Escalation",
        "period": "April – December 2018",
        "context": (
            "Escalating US tariffs on Chinese goods triggered retaliatory measures and "
            "broad EM stress. The yuan weakened close to the politically sensitive 7.00 "
            "level. Platinum was hit by fears of reduced Chinese auto demand and by the "
            "diesel emissions scandal depressing autocatalyst demand."
        ),
        "market_moves": {
            "us_10y_bps":       +30,    # modest rate rise before Q4 reversal
            "us_2y_bps":        +60,
            "usd_dxy_pct":      +8,
            "usdcnh_pct":       +9,     # CNH weakened from 6.25 to 6.95
            "usdjpy_pct":       +1,     # JPY relatively stable
            "chfjpy_pct":       -2,
            "equities_pct":     -20,    # S&P 500 Q4 2018 selloff
            "gold_pct":         -8,
            "platinum_pct":     -25,    # worst-performing major commodity
            "commodities_pct":  -15,
            "crypto_pct":       -80,    # crypto bear market coincided
            "vol_vix":          36,
        },
        "relevant_for": ["FX", "commodity", "equity", "crypto"],
    },

    # ----------------------------------------------------------
    # 4. 2013 Taper Tantrum
    # Most relevant for: long duration, EM FX, gold/platinum
    # ----------------------------------------------------------
    "2013_taper_tantrum": {
        "name": "2013 Taper Tantrum",
        "period": "May – September 2013",
        "context": (
            "Ben Bernanke's May 2013 suggestion that the Fed might taper QE purchases "
            "caused a sudden spike in US yields and a severe EM selloff. The move was "
            "entirely driven by a shift in Fed communication, not economic data, making "
            "it a clean example of duration risk without fundamental deterioration."
        ),
        "market_moves": {
            "us_10y_bps":       +100,   # 1.60% → 3.00% in ~4 months
            "us_2y_bps":        +45,
            "usd_dxy_pct":      +5,
            "usdcnh_pct":       +1,     # CNH relatively stable — PBOC managed
            "usdjpy_pct":       +10,    # JPY had already weakened on Abenomics
            "chfjpy_pct":       -8,     # CHF strengthened vs JPY
            "equities_pct":     -6,     # US equities modest dip, EM much worse
            "gold_pct":         -20,    # gold entered a prolonged bear market
            "platinum_pct":     -15,
            "commodities_pct":  -8,
            "crypto_pct":       None,
            "vol_vix":          21,
        },
        "relevant_for": ["rates", "commodity"],
    },

    # ----------------------------------------------------------
    # 5. 2020 COVID Crash
    # Most relevant for: crypto, equities, commodities, vol spike
    # ----------------------------------------------------------
    "2020_covid_crash": {
        "name": "2020 COVID-19 Market Crash",
        "period": "February – March 2020",
        "context": (
            "The fastest bear market in history. Global lockdowns collapsed demand "
            "across all asset classes simultaneously. Oil went briefly negative. "
            "The initial phase saw everything sold including gold — followed by "
            "a rapid Fed-driven recovery. Crypto initially crashed then rallied strongly."
        ),
        "market_moves": {
            "us_10y_bps":       -125,   # flight to quality; 10y to 0.50%
            "us_2y_bps":        -155,
            "usd_dxy_pct":      +8,     # USD initial safe-haven bid
            "usdcnh_pct":       +2,     # mild CNH weakness
            "usdjpy_pct":       -2,     # JPY briefly strengthened
            "chfjpy_pct":       +5,     # CHF safe-haven vs JPY
            "equities_pct":     -34,    # S&P 500 peak to trough in 33 days
            "gold_pct":         -12,    # initial selloff then +30% recovery
            "platinum_pct":     -35,    # industrial demand collapse
            "commodities_pct":  -35,    # oil most extreme
            "crypto_pct":       -50,    # bitcoin halved in one week
            "vol_vix":          85,
        },
        "relevant_for": ["equity", "commodity", "crypto", "rates", "vol"],
    },

    # ----------------------------------------------------------
    # 6. 2016 BOJ / SNB-Style Central Bank Surprise
    # Most relevant for: CHFJPY, JPY pairs, rates vol
    # ----------------------------------------------------------
    "2016_boj_policy_shock": {
        "name": "Central Bank Policy Surprise (BOJ / SNB analog)",
        "period": "2015 SNB peg removal / 2022 BOJ YCC tweak",
        "context": (
            "Sudden central bank policy reversals — like the SNB removing the EUR/CHF "
            "floor in January 2015 or the BOJ widening its yield curve control band in "
            "December 2022 — cause violent dislocations in JPY and CHF crosses. "
            "The key risk for CHFJPY is a BOJ hike surprising markets while the SNB "
            "holds, which compresses the cross sharply."
        ),
        "market_moves": {
            "us_10y_bps":       +15,    # modest spillover
            "us_2y_bps":        +10,
            "usd_dxy_pct":      -2,
            "usdcnh_pct":       -1,
            "usdjpy_pct":       -8,     # JPY strengthens sharply on BOJ hike
            "chfjpy_pct":       -10,    # CHF/JPY compresses (adverse for short)
            "equities_pct":     -5,     # modest risk-off
            "gold_pct":         +3,
            "platinum_pct":     +1,
            "commodities_pct":  -2,
            "crypto_pct":       -10,
            "vol_vix":          22,
        },
        "relevant_for": ["FX", "rates"],
    },

}


# ============================================================
# Theme → Scenario Relevance Map
# Used by select_scenarios() to match dominant portfolio
# themes to the most appropriate historical analogs.
# ============================================================

THEME_TO_SCENARIOS: dict[str, list[str]] = {
    "Rates & Duration": [
        "2022_rate_shock",
        "2013_taper_tantrum",
        "2020_covid_crash",
    ],
    "Foreign Exchange": [
        "2015_cny_devaluation",
        "2018_trade_war",
        "2016_boj_policy_shock",
    ],
    "Commodities": [
        "2018_trade_war",
        "2015_cny_devaluation",
        "2020_covid_crash",
    ],
    "Equities": [
        "2022_rate_shock",
        "2020_covid_crash",
        "2018_trade_war",
    ],
    "Digital Assets": [
        "2022_rate_shock",
        "2020_covid_crash",
        "2018_trade_war",
    ],
}


def select_scenarios(dominant_themes: list[str], n: int = 2) -> list[dict]:
    """
    Given the dominant portfolio themes (from _compute_risk_metrics),
    return the n most relevant scenarios from the library.

    Scores each scenario by how many of the dominant themes it covers,
    breaking ties by picking broader multi-theme scenarios first.

    Args:
        dominant_themes : list of theme strings, e.g. ["Rates & Duration", "Foreign Exchange"]
        n               : number of scenarios to return (default 2)

    Returns:
        List of scenario dicts, ready to inject into the risk agent prompt.

    Example:
        scenarios = select_scenarios(["Rates & Duration", "Commodities"])
        # → [2022_rate_shock, 2013_taper_tantrum]
    """
    scores: dict[str, int] = {key: 0 for key in STRESS_SCENARIOS}

    for theme in dominant_themes:
        for scenario_key in THEME_TO_SCENARIOS.get(theme, []):
            if scenario_key in scores:
                scores[scenario_key] += 1

    # Sort by score descending, then by number of relevant_for themes (broader = tiebreak)
    ranked = sorted(
        scores.items(),
        key=lambda x: (x[1], len(STRESS_SCENARIOS[x[0]]["relevant_for"])),
        reverse=True,
    )

    return [STRESS_SCENARIOS[key] for key, score in ranked[:n] if score > 0]


def format_scenario_for_prompt(scenario: dict) -> str:
    """
    Format a single scenario dict as a structured text block
    ready to inject into the risk agent user prompt.
    """
    moves = scenario["market_moves"]
    lines = [
        f"SCENARIO: {scenario['name']}",
        f"Period:   {scenario['period']}",
        f"Context:  {scenario['context']}",
        "",
        "Key market moves:",
    ]

    labels = {
        "us_10y_bps":       "  US 10y yield change",
        "us_2y_bps":        "  US 2y yield change",
        "usd_dxy_pct":      "  USD (DXY)",
        "usdcnh_pct":       "  USD/CNH",
        "usdjpy_pct":       "  USD/JPY",
        "chfjpy_pct":       "  CHF/JPY",
        "equities_pct":     "  Equities (S&P 500)",
        "gold_pct":         "  Gold",
        "platinum_pct":     "  Platinum",
        "commodities_pct":  "  Commodities (broad)",
        "crypto_pct":       "  Crypto (BTC)",
        "vol_vix":          "  Peak VIX",
    }

    for key, label in labels.items():
        val = moves.get(key)
        if val is None:
            continue
        if "bps" in key:
            lines.append(f"{label:<30} {val:>+6}bps")
        elif key == "vol_vix":
            lines.append(f"{label:<30} {val:>6}")
        else:
            lines.append(f"{label:<30} {val:>+6}%")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick preview of all scenarios and the selector
    print("=== SCENARIO LIBRARY ===\n")
    for key, s in STRESS_SCENARIOS.items():
        print(f"[{key}]  {s['name']}  ({s['period']})")
        print(f"  Relevant for: {', '.join(s['relevant_for'])}")
        print()

    print("\n=== SELECTOR TEST ===")
    print("Dominant themes: Rates & Duration, Foreign Exchange")
    selected = select_scenarios(["Rates & Duration", "Foreign Exchange"])
    for s in selected:
        print(f"\n{format_scenario_for_prompt(s)}")
