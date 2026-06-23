import os
import json
import random
from datetime import date, datetime, timedelta

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
load_dotenv()

app = FastAPI(title="Autonomous Closed-Loop Banking")
templates = Jinja2Templates(directory="templates")

TODAY = date.today()
INSURANCE_DUE_DAYS = 20
INSURANCE_AMOUNT = 18_000
SIMULATION_AMOUNT = 15_000

HABIT_SCENARIOS = ["gastro_creep", "subscription_trap", "weekend_micro", "overpaying"]

# Per-habit ranges for stochastic EV/Variance generation and display copy.
HABIT_META: dict[str, dict] = {
    "gastro_creep": {
        "name":    "Restaurace a volný čas – zvýšená útrata",
        "context": "Výdaje v kategorii Restaurace a volný čas vzrostly meziměsíčně o 35 %.",
        "detail":  "meziměsíční nárůst výdajů na stravování mimo domov o 35 %",
        "ev_min":  7_000,  "ev_max":  10_000,
        "var_min": 0.0010, "var_max": 0.0015,
    },
    "subscription_trap": {
        "name":    "Digitální služby a předplatné – neaktivní platby",
        "context": "Detekováno 6+ opakovaných plateb za digitální předplatné, která nejsou aktivně využívána.",
        "detail":  "6+ neaktivních měsíčních plateb za digitální předplatné",
        "ev_min":  12_000, "ev_max":  16_000,
        "var_min": 0.0020, "var_max": 0.0030,
    },
    "weekend_micro": {
        "name":    "Běžné nákupy a spotřeba – impulsivní výdaje",
        "context": "Vysoká frekvence neplánovaných nákupů spotřebního zboží mimo hlavní nákupní cyklus.",
        "detail":  "zvýšená frekvence neplánovaných spotřebních výdajů",
        "ev_min":  9_000,  "ev_max":  13_000,
        "var_min": 0.0015, "var_max": 0.0022,
    },
    "overpaying": {
        "name":    "Bydlení a služby – nadstandardní sazby",
        "context": "Fixní trvalé příkazy za energie a služby jsou o 20 % vyšší než průměr trhu.",
        "detail":  "fixní příkazy za bydlení a služby 20 % nad tržním průměrem",
        "ev_min":  7_500,  "ev_max":  11_000,
        "var_min": 0.0007, "var_max": 0.0011,
    },
}

# ---------------------------------------------------------------------------
# Expense donut – category definitions for Revolut-style doughnut chart
# ---------------------------------------------------------------------------
_EXPENSE_CATS = [
    {"key": "loans",   "label": "Splátky hypotéky a úvěrů", "pct": 28, "limit": 33, "color": "#3A4A5C"},
    {"key": "housing", "label": "Bydlení a služby",          "pct": 35, "limit": 35, "color": "#7B5EA7"},
    {"key": "daily",   "label": "Běžné nákupy a spotřeba",   "pct": 20, "limit": 18, "color": "#6E6E73"},
    {"key": "digital", "label": "Digitální služby a předpl.", "pct": 7, "limit":  5, "color": "#4A90D9"},
    {"key": "leisure", "label": "Restaurace a volný čas",    "pct": 10, "limit":  8, "color": "#FF6B35"},
]

_HABIT_TO_CAT: dict[str, tuple[str, int]] = {
    "gastro_creep":      ("leisure", 10),
    "subscription_trap": ("digital", 10),
    "weekend_micro":     ("daily",   10),
    "overpaying":        ("housing", 10),
}

# ---------------------------------------------------------------------------
# Risk profiles – MiFID II aligned, with distributional parameters
# ---------------------------------------------------------------------------
RISK_PROFILES: dict[str, dict] = {
    "konzervativni": {
        "label": "Konzervativní",
        "etf_pct": 0.20,
        "savings_pct": 0.80,
        "etf_label": "Stabilní ETF s nízkou volatilitou",
        "savings_label": "Dluhopisy a termínované vklady",
        "annual_return": 0.045,
        "expected_value": 4.5,
        "variance": 0.0015,
    },
    "vyvazeny": {
        "label": "Vyvážený",
        "etf_pct": 0.50,
        "savings_pct": 0.50,
        "etf_label": "Vyvážené akciové ETF (MSCI World)",
        "savings_label": "Dluhopisy a peněžní trh",
        "annual_return": 0.065,
        "expected_value": 6.5,
        "variance": 0.0080,
    },
    "dynamicky": {
        "label": "Dynamický",
        "etf_pct": 0.70,
        "savings_pct": 0.30,
        "etf_label": "Akciové ETF S&P 500 / MSCI World",
        "savings_label": "Ostatní trhy a alternativní aktiva",
        "annual_return": 0.08,
        "expected_value": 8.0,
        "variance": 0.0250,
    },
}


def _var_label_investment(var: float) -> str:
    """Human-readable market-volatility label for investment context."""
    if var < 0.005:
        return "s nízkou kolísavostí trhu"
    if var < 0.015:
        return "s mírnou kolísavostí trhu"
    return "s vyšší kolísavostí trhu (typické pro dynamický profil)"


# q1 answer → concrete year horizon (midpoint / common planning target)
_HORIZON_YEARS: dict[int, int] = {1: 3, 2: 5, 3: 10}

_HORIZON_LABEL: dict[int, str] = {
    1: "Krátkodobý cíl za 3 roky",
    2: "Střednědobý cíl za 5 let",
    3: "Dlouhodobý cíl za 10 let",
}


def _compute_investment_projection(
    monthly_pmt: int, years: int, risk_profile_key: str
) -> dict:
    """Future Value of regular monthly investment at 8 % p.a. with risk-adjusted σ.

    FV  = PMT × ((1+r)^n − 1) / r          (ordinary annuity, r = 8%/12, n = years×12)
    σ   = FV × √(profile_variance) × √years  (rough lognormal scaling)
    E[X] = FV  (deterministic at the fixed rate; variance captures market spread)
    """
    profile      = RISK_PROFILES.get(risk_profile_key, RISK_PROFILES["vyvazeny"])
    annual_rate  = profile["expected_value"]
    r            = (annual_rate / 100) / 12
    n            = years * 12
    fv           = int(monthly_pmt * ((1 + r) ** n - 1) / r)
    sigma        = int(fv * (profile["variance"] ** 0.5) * (years ** 0.5))
    return {
        "monthly_pmt":   monthly_pmt,
        "years":         years,
        "annual_rate":   annual_rate,
        "fv":            fv,
        "ev":            fv,
        "sigma":         sigma,
        "profile_label": profile["label"],
    }


def _var_label_spending(var: float) -> str:
    """Human-readable risk label for spending-habit context."""
    if var < 0.0012:
        return "nízká (stabilní, předvídatelný vzorec)"
    if var < 0.0022:
        return "mírná (občasné výkyvy)"
    return "vyšší (proměnlivý výdajový vzorec)"


def _score_to_profile(q1: int, q2: int, q3: int) -> tuple[str, int]:
    score = q1 + q2 + q3
    if q3 == 1:      # withdraw immediately → conservative regardless of experience
        return "konzervativni", score
    if q2 == 2:      # experienced + stay invested → dynamic
        return "dynamicky", score
    return "vyvazeny", score  # no experience + stay invested → balanced


# ---------------------------------------------------------------------------
# Daily tooltip label pools – per-habit and generic
# ---------------------------------------------------------------------------
_DAILY_LABEL_POOLS: dict[str, list[str]] = {
    "leisure": [
        "Restaurace a volný čas",
        "Výdaj: stravování mimo domov",
        "Neplánovaný výdaj: pohostinství a zábava",
    ],
    "digital": [
        "Digitální služby a předplatné",
        "Automatická platba: digitální předplatné",
        "Neaktivní předplatné – opakovaný výběr",
    ],
    "daily": [
        "Běžné nákupy a spotřeba",
        "Neplánovaný nákup: spotřební zboží",
        "Denní spotřeba: potravinářské zboží",
    ],
    "housing": [
        "Bydlení a služby – nadstandardní sazba",
        "Fixní platba: energetické a provozní služby",
        "Opakovaný výdaj: provoz domácnosti",
    ],
    "default": [
        "Běžné nákupy a spotřeba",
        "Denní výdaj: spotřební zboží",
        "Platba: potravinářské a drogistické zboží",
        "Výdaj: denní spotřeba domácnosti",
        "Nákup: smíšené spotřební zboží",
    ],
}


def _build_daily_tooltips(
    last60: list[dict], habit_scenario, rng: random.Random
) -> list[dict]:
    result: list[dict] = []
    for entry in last60:
        d       = date.fromisoformat(entry["date"])
        income  = entry["income"]
        net     = entry["net"]
        gastro  = entry.get("gastro", 0)
        subs    = entry.get("subscriptions", 0)
        w_micro = entry.get("weekend_micro", 0)

        if income > 0:
            result.append({"label": "Příchozí platba: pravidelná mzda", "delta": income})
            continue
        if d.day == 1:
            result.append({"label": "Bydlení a služby: nájemné", "delta": net})
            continue
        if d.day == 2:
            result.append({"label": "Splátka hypotéky a úvěrů RB", "delta": net})
            continue
        if d.day == 5:
            result.append({"label": "Bydlení a služby: energie", "delta": net})
            continue
        if d.day == 10 and subs > 0:
            result.append({"label": "Digitální služby a předplatné", "delta": net})
            continue
        if d.day == 18:
            result.append({"label": "Splátka úvěru RB (Minutová půjčka)", "delta": net})
            continue
        if net == 0:
            result.append({"label": "Žádné pohyby na účtu", "delta": 0})
            continue

        if habit_scenario == "gastro_creep" and gastro > 0:
            pool = _DAILY_LABEL_POOLS["leisure"]
        elif habit_scenario == "subscription_trap" and d.weekday() in (0, 2, 4):
            pool = _DAILY_LABEL_POOLS["digital"]
        elif habit_scenario == "weekend_micro" and w_micro > 0:
            pool = _DAILY_LABEL_POOLS["daily"]
        elif habit_scenario == "overpaying" and d.weekday() in (1, 3):
            pool = _DAILY_LABEL_POOLS["housing"]
        else:
            pool = _DAILY_LABEL_POOLS["default"]

        result.append({"label": rng.choice(pool), "delta": net})

    return result


def _build_prediction_tooltips(prediction: list[dict]) -> list[dict]:
    result: list[dict] = []
    for entry in prediction:
        d      = date.fromisoformat(entry["date"])
        income = entry["income"]
        net    = entry["net"]

        if income > 0:
            result.append({"label": "Predikce: příchozí mzda", "delta": income})
        elif entry.get("is_stress"):
            result.append({"label": "Predikce: platba roční pojistky", "delta": -INSURANCE_AMOUNT})
        elif entry.get("is_agent"):
            result.append({"label": "Predikce: AI Autopilot – přesun do ETF portfolia", "delta": -SIMULATION_AMOUNT})
        elif d.day == 1:
            result.append({"label": "Predikce: nájemné + denní výdaje", "delta": net})
        elif d.day == 5:
            result.append({"label": "Predikce: záloha energie + denní výdaje", "delta": net})
        elif d.day == 10:
            result.append({"label": "Predikce: předplatné + denní výdaje", "delta": net})
        else:
            result.append({"label": "Predikce: odhadovaný denní výdaj", "delta": -675})

    return result


# ---------------------------------------------------------------------------
# Mock history – deterministic daily amounts (seed 42), variable start balance
# ---------------------------------------------------------------------------
def _build_history(
    initial_balance: int = 52_000,
    mortgage_payment: int = 18_000,
    mini_loan_payment: int = 4_000,
    income_rng=None,
) -> list[dict]:
    if income_rng is None:
        income_rng = random.Random(43)

    days = [(i, TODAY - timedelta(days=i)) for i in range(180, 0, -1)]

    # Pass 1 – tally total expenses per calendar month (mirror seed for daily amounts)
    month_exp: dict[tuple, int] = {}
    _temp = random.Random(42)
    for i, d in days:
        exp = _temp.randint(400, 950) if i <= 60 else _temp.randint(300, 850)
        if d.day == 1:  exp += 22_000
        if d.day == 2:  exp += mortgage_payment
        if d.day == 5:  exp += 4_500
        if d.day == 10: exp += 1_500 if i <= 60 else 1_200
        if d.day == 18: exp += mini_loan_payment
        key = (d.year, d.month)
        month_exp[key] = month_exp.get(key, 0) + exp

    # Generate variable monthly income with guardrail: income ≥ expenses + random buffer
    month_income: dict[tuple, int] = {}
    for key in sorted(month_exp):
        base   = income_rng.randint(55_000, 75_000)
        buffer = income_rng.randint(10_000, 25_000)
        month_income[key] = max(base, month_exp[key] + buffer)

    # Pass 2 – build entries using pre-computed incomes and same daily-amount sequence
    rng     = random.Random(42)
    entries = []
    balance = initial_balance

    for i, d in days:
        key           = (d.year, d.month)
        income        = month_income.get(key, 0) if d.day == 15 else 0
        expense       = 0
        gastro        = 0
        subscriptions = 0
        weekend_micro = 0
        mortgage      = 0
        mini_loan     = 0

        if d.day == 1:
            expense += 22_000
        if d.day == 2:
            mortgage  = mortgage_payment
            expense  += mortgage
        if d.day == 5:
            expense += 4_500
        if d.day == 10:
            subscriptions = 1_500 if i <= 60 else 1_200
            expense += subscriptions
        if d.day == 18:
            mini_loan  = mini_loan_payment
            expense   += mini_loan

        daily    = rng.randint(400, 950) if i <= 60 else rng.randint(300, 850)
        expense += daily

        if d.weekday() in (2, 5):
            gastro = round(daily * 0.55)
        if d.weekday() in (5, 6):
            weekend_micro = round(daily * 0.45)

        net      = income - expense
        balance += net
        entries.append({
            "date":          d.isoformat(),
            "income":        income,
            "expense":       expense,
            "net":           net,
            "balance":       balance,
            "gastro":        gastro,
            "subscriptions": subscriptions,
            "weekend_micro": weekend_micro,
            "mortgage":      mortgage,
            "mini_loan":     mini_loan,
            "cat_loans":     mortgage + mini_loan,
            "cat_housing":   (22_000 if d.day == 1 else 0) + (4_500 if d.day == 5 else 0),
            "cat_digital":   subscriptions,
            "cat_leisure":   gastro,
            "cat_daily":     daily - gastro,
        })
    return entries


def _build_prediction(
    current_balance: float,
    mortgage_payment: int = 18_000,
    mini_loan_payment: int = 4_000,
    pred_income: int = 65_000,
) -> list[dict]:
    entries = []
    balance = current_balance

    for i in range(1, 31):
        d = TODAY + timedelta(days=i)
        income = 0
        expense = 0
        mortgage = 0
        mini_loan = 0

        if d.day == 15:
            income += pred_income
        if d.day == 1:
            expense += 22_000
        if d.day == 2:
            mortgage = mortgage_payment
            expense += mortgage
        if d.day == 5:
            expense += 4_500
        if d.day == 10:
            expense += 1_500
        if d.day == 18:
            mini_loan = mini_loan_payment
            expense += mini_loan

        # AI Autopilot redirects surplus to ETF on day 22 of the prediction window
        is_agent = (i == 22)
        if is_agent:
            expense += SIMULATION_AMOUNT

        # AI Autopilot cuts variable habit spending by ~33 % (450 vs historic 675)
        daily_var = 450
        expense  += daily_var
        leisure   = round(daily_var * 0.30)
        daily_buy = daily_var - leisure

        net = income - expense
        balance += net
        surplus = max(0.0, balance - 30_000)
        entries.append({
            "date":      d.isoformat(),
            "income":    income,
            "expense":   expense,
            "net":       net,
            "balance":   balance,
            "surplus":   surplus,
            "is_stress": False,
            "is_agent":  is_agent,
            "mortgage":  mortgage,
            "mini_loan": mini_loan,
            "cat_loans":   mortgage + mini_loan,
            "cat_housing": (22_000 if d.day == 1 else 0) + (4_500 if d.day == 5 else 0),
            "cat_digital": (1_500 if d.day == 10 else 0),
            "cat_leisure": leisure,
            "cat_daily":   daily_buy,
        })
    return entries


# ---------------------------------------------------------------------------
# Stochastic habit helpers
# ---------------------------------------------------------------------------
def _pick_habit(rng: random.Random) -> dict:
    """Randomly choose one habit and generate its EV/Variance within realistic ranges."""
    scenario = rng.choice(HABIT_SCENARIOS)
    meta     = HABIT_META[scenario]
    return {
        "scenario": scenario,
        "ev":       rng.randint(meta["ev_min"], meta["ev_max"]),
        "variance": round(rng.uniform(meta["var_min"], meta["var_max"]), 4),
    }



# ---------------------------------------------------------------------------
# Localisation – all translated strings (months, categories, UI copy, habits)
# ---------------------------------------------------------------------------
_T: dict = {
    "cz": {
        "months": ["Leden","Únor","Březen","Duben","Květen","Červen",
                   "Červenec","Srpen","Září","Říjen","Listopad","Prosinec"],
        "cats": {
            "loans":   "Splátky hypotéky a úvěrů",
            "housing": "Bydlení a služby",
            "daily":   "Běžné nákupy a spotřeba",
            "digital": "Digitální služby a předpl.",
            "leisure": "Restaurace a volný čas",
        },
        "story": {
            "income_label":  "Výplata",
            "expense_label": "Výdajový vrchol",
            "loans_label":   "Pravidelné závazky 🏠",
            "agent_label":   "AI Autopilot",
            "income_text":   "Na účet dorazila pravidelná mzda {salary}. Zůstatek po připsání: {balance}.",
            "expense_text":  "Výdajový vrchol {amount}: nájemné, energie a výdaj [{habit}]. Zůstatek: {balance}.",
            "loans_text":    "Odešlo na splátky: {total}. Splátka hypotéky: {mortgage}, Minutová půjčka: {loan}.",
            "smart_rate":    "[Smart Rate 📉] Raiffeisenbank uplatňuje slevu 0,2 % p.a. na vaši hypotéku.",
            "agent_text":    "AI Autopilot zkrotil výdaje o ~33 %. Přebytek {amount} přesměrován do ETF. Projekce ETF: {etf}.",
            "success_fee":   "Success fee 10 % ze zjištěných úspor. Ušetřeno: {savings}, poplatek RB: {fee}.",
        },
        "habits": {
            "gastro_creep":      {"name": "Restaurace a volný čas – zvýšená útrata",            "detail": "meziměsíční nárůst výdajů na stravování mimo domov o 35 %"},
            "subscription_trap": {"name": "Digitální služby a předplatné – neaktivní platby",   "detail": "6+ neaktivních měsíčních plateb za digitální předplatné"},
            "weekend_micro":     {"name": "Běžné nákupy a spotřeba – impulsivní výdaje",        "detail": "zvýšená frekvence neplánovaných spotřebních výdajů"},
            "overpaying":        {"name": "Bydlení a služby – nadstandardní sazby",             "detail": "fixní příkazy za bydlení a služby 20 % nad tržním průměrem"},
        },
        "horizon": {1: "Krátkodobý cíl za 3 roky", 2: "Střednědobý cíl za 5 let", 3: "Dlouhodobý cíl za 10 let"},
    },
    "ua": {
        "months": ["Січень","Лютий","Березень","Квітень","Травень","Червень",
                   "Липень","Серпень","Вересень","Жовтень","Листопад","Грудень"],
        "cats": {
            "loans":   "Виплати іпотеки та кредитів",
            "housing": "Житло та комунальні послуги",
            "daily":   "Повсякденні покупки",
            "digital": "Цифрові послуги та підписки",
            "leisure": "Ресторани та дозвілля",
        },
        "story": {
            "income_label":  "Зарплата",
            "expense_label": "Пік витрат",
            "loans_label":   "Регулярні зобов'язання 🏠",
            "agent_label":   "AI Autopilot",
            "income_text":   "На рахунок надійшла зарплата {salary}. Залишок після зарахування: {balance}.",
            "expense_text":  "Пік витрат {amount}: оренда, комунальні та неефективні витрати [{habit}]. Залишок: {balance}.",
            "loans_text":    "Відправлено на виплати: {total}. Іпотека: {mortgage}, Мікрокредит: {loan}.",
            "smart_rate":    "[Smart Rate 📉] Raiffeisenbank надає знижку 0,2 % p.a. на вашу іпотеку.",
            "agent_text":    "AI Autopilot скоротив витрати на ~33 %. Надлишок {amount} переведено в ETF. Прогноз ETF: {etf}.",
            "success_fee":   "Success fee 10 % з виявлених заощаджень. Заощаджено: {savings}, комісія: {fee}.",
        },
        "habits": {
            "gastro_creep":      {"name": "Ресторани та дозвілля – підвищені витрати",              "detail": "зростання витрат на харчування поза домом на 35 %"},
            "subscription_trap": {"name": "Цифрові послуги – неактивні підписки",                  "detail": "6+ неактивних щомісячних платежів за цифрові підписки"},
            "weekend_micro":     {"name": "Повсякденні покупки – імпульсивні витрати",              "detail": "підвищена частота незапланованих покупок"},
            "overpaying":        {"name": "Житло та послуги – завищені тарифи",                     "detail": "комунальні платежі на 20 % вище середньоринкових"},
        },
        "horizon": {1: "Короткострокова мета на 3 роки", 2: "Середньострокова мета на 5 років", 3: "Довгострокова мета на 10 років"},
    },
    "sk": {
        "months": ["Január","Február","Marec","Apríl","Máj","Jún",
                   "Júl","August","September","Október","November","December"],
        "cats": {
            "loans":   "Splátky hypotéky a úverov",
            "housing": "Bývanie a služby",
            "daily":   "Bežné nákupy a spotreba",
            "digital": "Digitálne služby a predpl.",
            "leisure": "Reštaurácie a voľný čas",
        },
        "story": {
            "income_label":  "Výplata",
            "expense_label": "Výdajový vrchol",
            "loans_label":   "Pravidelné záväzky 🏠",
            "agent_label":   "AI Autopilot",
            "income_text":   "Na účet prišla pravidelná mzda {salary}. Zostatok po pripísaní: {balance}.",
            "expense_text":  "Výdajový vrchol {amount}: nájomné, energie a výdavok [{habit}]. Zostatok: {balance}.",
            "loans_text":    "Odišlo na splátky: {total}. Splátka hypotéky: {mortgage}, Pôžička: {loan}.",
            "smart_rate":    "[Smart Rate 📉] Raiffeisenbank uplatňuje zľavu 0,2 % p.a. na vašu hypotéku.",
            "agent_text":    "AI Autopilot znížil výdavky o ~33 %. Prebytok {amount} presmerovaný do ETF. Projekcia ETF: {etf}.",
            "success_fee":   "Success fee 10 % z odhalených úspor. Ušetrené: {savings}, poplatok RB: {fee}.",
        },
        "habits": {
            "gastro_creep":      {"name": "Reštaurácie a voľný čas – zvýšená útrata",             "detail": "medziměsačný nárast výdavkov na stravovanie mimo domova o 35 %"},
            "subscription_trap": {"name": "Digitálne služby a predplatné – neaktívne platby",     "detail": "6+ neaktívnych mesačných platieb za digitálne predplatné"},
            "weekend_micro":     {"name": "Bežné nákupy a spotreba – impulzívne výdavky",         "detail": "zvýšená frekvencia neplánovaných spotrebiteľských výdavkov"},
            "overpaying":        {"name": "Bývanie a služby – nadštandardné sadzby",              "detail": "fixné príkazy za bývanie 20 % nad trhovou úrovňou"},
        },
        "horizon": {1: "Krátkodobý cieľ za 3 roky", 2: "Strednodobý cieľ za 5 rokov", 3: "Dlhodobý cieľ za 10 rokov"},
    },
    "vn": {
        "months": ["Tháng 1","Tháng 2","Tháng 3","Tháng 4","Tháng 5","Tháng 6",
                   "Tháng 7","Tháng 8","Tháng 9","Tháng 10","Tháng 11","Tháng 12"],
        "cats": {
            "loans":   "Trả góp thế chấp và vay",
            "housing": "Nhà ở và dịch vụ",
            "daily":   "Mua sắm hàng ngày",
            "digital": "Dịch vụ số và đăng ký",
            "leisure": "Nhà hàng và giải trí",
        },
        "story": {
            "income_label":  "Lương",
            "expense_label": "Đỉnh chi tiêu",
            "loans_label":   "Nghĩa vụ thường xuyên 🏠",
            "agent_label":   "AI Autopilot",
            "income_text":   "Tài khoản nhận lương {salary}. Số dư sau khi nhận: {balance}.",
            "expense_text":  "Đỉnh chi tiêu {amount}: tiền thuê, điện nước và chi tiêu kém hiệu quả [{habit}]. Số dư: {balance}.",
            "loans_text":    "Đã thanh toán: {total}. Trả góp thế chấp: {mortgage}, Khoản vay: {loan}.",
            "smart_rate":    "[Smart Rate 📉] Raiffeisenbank giảm 0,2 % p.a. lãi suất thế chấp của bạn.",
            "agent_text":    "AI Autopilot giảm chi tiêu ~33 %. {amount} được chuyển vào ETF. Dự báo ETF: {etf}.",
            "success_fee":   "Success fee 10 % từ khoản tiết kiệm phát hiện. Tiết kiệm: {savings}, phí: {fee}.",
        },
        "habits": {
            "gastro_creep":      {"name": "Nhà hàng và giải trí – chi tiêu tăng cao",             "detail": "chi tiêu ăn ngoài tăng 35% so với tháng trước"},
            "subscription_trap": {"name": "Dịch vụ số – đăng ký không hoạt động",                "detail": "6+ khoản đăng ký kỹ thuật số không được sử dụng"},
            "weekend_micro":     {"name": "Mua sắm hàng ngày – chi tiêu bốc đồng",               "detail": "tần suất mua sắm tự phát cao bất thường"},
            "overpaying":        {"name": "Nhà ở và dịch vụ – mức giá quá cao",                  "detail": "hóa đơn tiện ích cao hơn 20% mức thị trường"},
        },
        "horizon": {1: "Mục tiêu ngắn hạn 3 năm", 2: "Mục tiêu trung hạn 5 năm", 3: "Mục tiêu dài hạn 10 năm"},
    },
    "ru": {
        "months": ["Январь","Февраль","Март","Апрель","Май","Июнь",
                   "Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"],
        "cats": {
            "loans":   "Выплаты ипотеки и кредитов",
            "housing": "Жильё и коммунальные услуги",
            "daily":   "Повседневные покупки",
            "digital": "Цифровые услуги и подписки",
            "leisure": "Рестораны и досуг",
        },
        "story": {
            "income_label":  "Зарплата",
            "expense_label": "Пик расходов",
            "loans_label":   "Регулярные обязательства 🏠",
            "agent_label":   "AI Autopilot",
            "income_text":   "На счёт поступила зарплата {salary}. Остаток после зачисления: {balance}.",
            "expense_text":  "Пик расходов {amount}: аренда, коммунальные и неэффективные расходы [{habit}]. Остаток: {balance}.",
            "loans_text":    "Отправлено на выплаты: {total}. Выплата ипотеки: {mortgage}, Микрокредит: {loan}.",
            "smart_rate":    "[Smart Rate 📉] Raiffeisenbank предоставляет скидку 0,2 % p.a. на вашу ипотеку.",
            "agent_text":    "AI Autopilot снизил расходы на ~33 %. Излишек {amount} направлен в ETF. Прогноз ETF: {etf}.",
            "success_fee":   "Success fee 10 % с выявленных сбережений. Сэкономлено: {savings}, комиссия: {fee}.",
        },
        "habits": {
            "gastro_creep":      {"name": "Рестораны и досуг – повышенные расходы",              "detail": "рост расходов на питание вне дома на 35 % за месяц"},
            "subscription_trap": {"name": "Цифровые услуги – неактивные подписки",               "detail": "6+ неактивных ежемесячных платежей за цифровые подписки"},
            "weekend_micro":     {"name": "Повседневные покупки – импульсивные траты",           "detail": "повышенная частота незапланированных покупок"},
            "overpaying":        {"name": "Жильё и услуги – завышенные тарифы",                  "detail": "платежи за жильё на 20 % выше рыночного уровня"},
        },
        "horizon": {1: "Краткосрочная цель на 3 года", 2: "Среднесрочная цель на 5 лет", 3: "Долгосрочная цель на 10 лет"},
    },
}


def _t(lang: str, *keys):
    """Get translation with Czech fallback. Keys navigate nested dicts."""
    for source in (_T.get(lang, {}), _T["cz"]):
        node = source
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node is not None:
            return node
    return ""


def _build_detected_habit(habit: dict, lang: str = "cz") -> list[dict]:
    """Build the detected_habits payload from a habit_info dict (no log entry)."""
    h_t = _t(lang, "habits", habit["scenario"])
    meta = HABIT_META[habit["scenario"]]
    return [{
        "key":      habit["scenario"],
        "name":     h_t.get("name", meta["name"])   if isinstance(h_t, dict) else meta["name"],
        "detail":   h_t.get("detail", meta["detail"]) if isinstance(h_t, dict) else meta["detail"],
        "ev_loss":  habit["ev"],
        "variance": habit["variance"],
    }]


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_INITIAL = {
    "savings_balance": 5_000.0,
    "portfolio_value": 12_000.0,
    "portfolio_units": 80.0,
    "etf_price": 150.0,
}

_state: dict = {
    **_INITIAL,
    "checking_balance": 0.0,
    "agent_log": [],
    "detected_habits": None,
    "habit_scenario": "gastro_creep",
    "habit_info": None,
}


def _init_state(initial_balance: int = 52_000, rng=None) -> None:
    if rng is None:
        rng = random.Random(int.from_bytes(os.urandom(8), "big"))
    mortgage_payment  = rng.randint(15_000, 22_000)
    mini_loan_payment = rng.randint(3_000, 5_000)
    _state["mortgage_payment"]  = mortgage_payment
    _state["mini_loan_payment"] = mini_loan_payment

    # Build history – income_rng=rng makes monthly incomes vary per reset
    history = _build_history(initial_balance, mortgage_payment, mini_loan_payment, income_rng=rng)
    _state["history"]          = history
    _state["checking_balance"] = history[-1]["balance"]

    # Pre-compute prediction month's total display expenses for the income guardrail
    pred_exp_display = sum(
        450  # AI-optimized daily variable (SIMULATION_AMOUNT excluded from display)
        + (22_000          if (TODAY + timedelta(days=i)).day == 1  else 0)
        + (mortgage_payment if (TODAY + timedelta(days=i)).day == 2  else 0)
        + (4_500           if (TODAY + timedelta(days=i)).day == 5  else 0)
        + (1_500           if (TODAY + timedelta(days=i)).day == 10 else 0)
        + (mini_loan_payment if (TODAY + timedelta(days=i)).day == 18 else 0)
        for i in range(1, 31)
    )
    pred_base   = rng.randint(55_000, 75_000)
    pred_buffer = rng.randint(10_000, 25_000)
    pred_income = max(pred_base, pred_exp_display + pred_buffer)

    _state["prediction"] = _build_prediction(
        _state["checking_balance"], mortgage_payment, mini_loan_payment, pred_income
    )
    habit = _pick_habit(rng)
    _state["habit_info"]       = habit
    _state["habit_scenario"]   = habit["scenario"]
    _state["detected_habits"]  = _build_detected_habit(habit)
    _state["monthly_expense"]  = rng.randint(30_000, 60_000)
    _state["monthly_salary"]   = pred_income
    _state["history_tooltips"]    = _build_daily_tooltips(history[-60:], habit["scenario"], rng)
    _state["prediction_tooltips"] = _build_prediction_tooltips(_state["prediction"])


_init_state()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(action: str, message: str, amount: float = 0.0) -> None:
    _state["agent_log"].append({"ts": _ts(), "action": action, "amount": amount, "message": message})


def transfer_to_savings(amount: float) -> dict:
    if amount <= 0:
        return {"success": False, "reason": "Amount must be positive."}
    if amount > _state["checking_balance"]:
        return {"success": False, "reason": "Insufficient funds."}
    _state["checking_balance"] -= amount
    _state["savings_balance"] += amount
    msg = (
        f"Převod {amount:,.0f} Kč na spořicí účet dokončen. "
        f"Nový zůstatek běžného účtu: {_state['checking_balance']:,.0f} Kč."
    )
    _log("transfer_to_savings", msg, amount)
    return {"success": True, "new_checking": _state["checking_balance"], "new_savings": _state["savings_balance"]}


def invest_funds(amount: float) -> dict:
    if amount <= 0:
        return {"success": False, "reason": "Amount must be positive."}
    if amount > _state["checking_balance"]:
        return {"success": False, "reason": "Insufficient funds."}
    units = amount / _state["etf_price"]
    _state["checking_balance"] -= amount
    _state["portfolio_units"] += units
    _state["portfolio_value"] = _state["portfolio_units"] * _state["etf_price"]
    msg = (
        f"Investováno {amount:,.0f} Kč → {units:.2f} ETF j. @ {_state['etf_price']:.2f} Kč. "
        f"Portfolio celkem: {_state['portfolio_value']:,.0f} Kč."
    )
    _log("invest_funds", msg, amount)
    return {"success": True, "units_bought": units, "portfolio_value": _state["portfolio_value"]}


TOOLS = [
    {
        "name": "transfer_to_savings",
        "description": "Převede částku z běžného účtu na spořicí účet.",
        "input_schema": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]},
    },
    {
        "name": "invest_funds",
        "description": "Investuje částku z běžného účtu do ETF portfolia.",
        "input_schema": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]},
    },
]


def _dispatch_tool(name: str, inp: dict) -> dict:
    if name == "transfer_to_savings":
        return transfer_to_savings(inp["amount"])
    if name == "invest_funds":
        return invest_funds(inp["amount"])
    return {"error": "Unknown tool"}


# ---------------------------------------------------------------------------
# Spending-habit log writer (reads pre-generated habit_info from state)
# ---------------------------------------------------------------------------
def _detect_spending_habits(risk_profile: str) -> list[dict]:
    """Log the active habit warning using the EV/Variance already generated at reset time."""
    habit = _state.get("habit_info")
    if not habit:
        _state["detected_habits"] = []
        return []

    meta     = HABIT_META[habit["scenario"]]
    ev       = habit["ev"]
    variance = habit["variance"]
    profile  = RISK_PROFILES.get(risk_profile, RISK_PROFILES["vyvazeny"])

    _log(
        "spending_warning",
        f"[Analýza chování ⚠️] Detekován skrytý únik peněz: '{meta['name']}'. "
        f"{meta['context']} "
        f"Očekávaná roční ztráta: {ev:,} Kč při {_var_label_spending(variance)} riziku. "
        f"Agent doporučuje okamžité přesměrování ušetřené částky do {profile['etf_label']}. "
        f"(Matematický model: E[ZTRÁTA] = {ev:,} Kč, Var(X) = {variance:.4f})",
    )
    detected = _build_detected_habit(habit)
    _state["detected_habits"] = detected
    return detected


# ---------------------------------------------------------------------------
# Simulation fallback (no API key)
# ---------------------------------------------------------------------------
def _run_fallback(risk_profile: str = "vyvazeny") -> str:
    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["vyvazeny"])
    amount = SIMULATION_AMOUNT
    etf_amount = round(amount * profile["etf_pct"])
    savings_amount = amount - etf_amount

    _log(
        "prediction",
        f"[Predikce] Detekován bezpečný dlouhodobý přebytek {amount:,} Kč "
        "po pokrytí všech měsíčních závazků (hypotéka, úvěry, bydlení).",
    )

    _detect_spending_habits(risk_profile)

    _log("execution", f"[Exekuce] Spouštím autonomní přesun {amount:,} Kč z běžného účtu.", amount)

    if amount <= _state["checking_balance"]:
        if etf_amount > 0:
            units = etf_amount / _state["etf_price"]
            _state["checking_balance"] -= etf_amount
            _state["portfolio_units"] += units
            _state["portfolio_value"] = _state["portfolio_units"] * _state["etf_price"]
        if savings_amount > 0:
            _state["checking_balance"] -= savings_amount
            _state["savings_balance"] += savings_amount

    ev  = profile["expected_value"]
    var = profile["variance"]

    if risk_profile == "dynamicky":
        sp500_amt = round(amount * 0.70)
        msci_amt  = amount - sp500_amt
        wm_msg = (
            f"[Wealth Management 📈] Detekoval jsem dlouhý investiční horizont a vysokou odvahu "
            f"riskovat. Vašich {amount:,} Kč jsem proto rozdělil takto:\n"
            f"- {sp500_amt:,} Kč (70 %) investuji do iShares Core S&P 500 ETF. Tento fond nakupuje "
            f"akcie 500 největších amerických firem. Vaše peníze tak nyní spoluvlastní giganty jako "
            f"Apple, Microsoft, Nvidia a Google. Je to motor pro maximální dlouhodobý růst.\n"
            f"- {msci_amt:,} Kč (30 %) dávám do Amundi MSCI World ETF. Tento fond investuje do "
            f"tisíců firem napříč Evropou a Japonskem, abychom nespoléhali jen na Ameriku a snížili riziko.\n"
            f"Předpokládaný roční výnos: {ev:.1f} % p.a. {_var_label_investment(var)}. "
            f"(Model: E[X] = {ev:.1f}%, Var(X) = {var:.4f})"
        )
    elif risk_profile == "vyvazeny":
        eq_amt   = round(amount * 0.50)
        bond_amt = amount - eq_amt
        wm_msg = (
            f"[Wealth Management ⚖️] Zvolil jste zlatou střední cestu (horizont 3–7 let). "
            f"Částku {amount:,} Kč dělím přesně na polovinu:\n"
            f"- {eq_amt:,} Kč (50 %) investuji do Vanguard FTSE All-World ETF. Tento fond obsahuje "
            f"mix akcií z celého světa, který vám zajistí stabilní růst.\n"
            f"- {bond_amt:,} Kč (50 %) posílám do iShares Core Global Aggregate Bond ETF. To je "
            f"balík bezpečných vládních a firemních dluhopisů. Pokud akciové trhy začnou klesat, "
            f"tyto dluhopisy budou fungovat jako polštář a ochrání váš účet před velkými propady.\n"
            f"Předpokládaný roční výnos: {ev:.1f} % p.a. {_var_label_investment(var)}. "
            f"(Model: E[X] = {ev:.1f}%, Var(X) = {var:.4f})"
        )
    else:  # konzervativni
        bond_amt   = round(amount * 0.80)
        liquid_amt = amount - bond_amt
        wm_msg = (
            f"[Wealth Management 🛡️] Vaší prioritou je bezpečí a ochrana před inflací bez riskování. "
            f"Částku {amount:,} Kč ukládám takto:\n"
            f"- {bond_amt:,} Kč (80 %) investuji do iShares EUR Ultrashort Bond ETF. Jde o nejbezpečnější "
            f"krátkodobé evropské dluhopisy s minimálním kolísáním ceny.\n"
            f"- {liquid_amt:,} Kč (20 %) přesouvám přímo na váš Spořicí účet v Raiffeisenbank. "
            f"Tyto peníze nikam neuzamykám, zůstávají vám plně po ruce jako okamžitá rezerva, "
            f"ale úročí se lepším úrokem.\n"
            f"Předpokládaný roční výnos: {ev:.1f} % p.a. {_var_label_investment(var)}. "
            f"(Model: E[X] = {ev:.1f}%, Var(X) = {var:.4f})"
        )

    _log("wealth_management", wm_msg)

    savings_bal      = _state["savings_balance"]
    savings_interest = round(savings_bal * 0.035)
    savings_tax      = round(savings_interest * 0.15)
    _log(
        "banking_fees",
        f"[Transparentnost poplatků RB] "
        f"Srážková daň z úroků (15 %): {savings_tax:,} Kč automaticky odečtena ze spořicího bonusu {savings_interest:,} Kč. "
        f"Vedení prémiového účtu: 0 Kč (aktivováno zdarma – splněny podmínky pravidelného příjmu).",
    )

    _log(
        "compliance",
        f"[Compliance ⚖️] Kontrola MiFID II profilu: Validní. "
        f"Zvolený profil '{profile['label']}' odpovídá výsledku dotazníku. "
        f"Režim Kopilot vyžaduje push autorizaci v RB klíči.",
    )

    habit      = _state.get("habit_info")
    habit_ev   = habit["ev"]   if habit else 0
    habit_var  = habit["variance"] if habit else 0.0
    _log(
        "risk_engine",
        f"[Risk Engine 🧠] Propočet Expected Value a Variance: "
        f"E[ZTRÁTA_HABIT] = {habit_ev:,} Kč/rok, Var(X) = {habit_var:.4f}. "
        f"Investiční model: E[X] = {ev:.1f} % p.a., Var(X) = {var:.4f}. "
        f"Úspěšně kalibrováno na historickou volatilitu trhu pod kódem RB-ALM-01.",
    )

    fixed_monthly = _state.get("mortgage_payment", 18_000) + _state.get("mini_loan_payment", 4_000) + 22_000 + 4_500
    reserve_target = fixed_monthly * 3
    current_reserve = _state["savings_balance"]
    reserve_ok = "Autonomie je bezpečná." if current_reserve >= reserve_target else f"Doporučeno navýšit rezervu na {reserve_target:,} Kč."
    _log(
        "liquidity_guard",
        f"[Liquidity Guard 🛡️] Rezerva na spořicím účtu: {current_reserve:,.0f} Kč. "
        f"Minimální cíl (3× fixní výdaje = {reserve_target:,} Kč): "
        f"{'splněn' if current_reserve >= reserve_target else 'NESPLNĚN'}. {reserve_ok}",
    )

    _log(
        "biz_model",
        f"[Byznys model 💼] Alokace {etf_amount:,} Kč do ETF realizována přes platformu RB Invest "
        f"(Management Fee 0,75 % p.a. = {round(etf_amount * 0.0075):,} Kč/rok, schváleno dle MiFID II). "
        f"Success fee za optimalizaci výdajů: 10 % ze zjištěných úspor.",
    )

    _log(
        "credit_risk",
        f"[Kreditní riziko 📊] Index finančního zdraví klienta stoupl na 94/100. "
        f"Riziko delikvence kleslo o 15 % vlivem pravidelného investičního chování a optimalizace výdajů. "
        f"Skóre přepočteno na základě cash flow za posledních 6 měsíců (model RB-CREDIT-02).",
    )

    return f"Simulace dokončena. Profil: {profile['label']}. Přesunuto {amount:,} Kč."


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------
def run_agent(risk_profile: str = "vyvazeny") -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _run_fallback(risk_profile)

    prediction = _state["prediction"]
    safe_surplus = min(e["surplus"] for e in prediction[:15]) if prediction else 0.0
    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["vyvazeny"])
    mortgage  = _state.get("mortgage_payment", 18_000)
    mini_loan = _state.get("mini_loan_payment", 4_000)

    system_prompt = (
        f"Jsi autonomní finanční agent pro systém Closed-Loop Banking. "
        f"Rizikový profil uživatele (MiFID II): {profile['label']} "
        f"(ETF {round(profile['etf_pct']*100)} %, spoření {round(profile['savings_pct']*100)} %). "
        "Analyzuj cash flow, detekuj neefektivní výdaje a proveď optimalizaci pomocí nástrojů. "
        "Odpovídej česky, stručně a přesně."
    )

    user_msg = (
        f"Běžný účet: {_state['checking_balance']:,.0f} Kč\n"
        f"Spořicí účet: {_state['savings_balance']:,.0f} Kč\n"
        f"Portfolio (ETF): {_state['portfolio_value']:,.0f} Kč\n"
        f"Měsíční fixní závazky: splátka hypotéky {mortgage:,} Kč + půjčka {mini_loan:,} Kč\n"
        f"Odhadovaný přebytek před závazky: {safe_surplus:,.0f} Kč\n\n"
        "Proveď autonomní rozhodnutí."
    )

    try:
        _detect_spending_habits(risk_profile)
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": user_msg}]

        for _ in range(5):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            for block in response.content:
                if hasattr(block, "text") and block.text:
                    _log("agent_thought", block.text)

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        final_text = " ".join(
            block.text for block in response.content if hasattr(block, "text") and block.text
        )
        return final_text or "Agent dokončil analýzu."

    except Exception:
        return _run_fallback(risk_profile)


# ---------------------------------------------------------------------------
# Chart story – single source of truth: all values read from the same arrays
# the chart sends to the frontend, so cards always match the graph exactly.
# ---------------------------------------------------------------------------
def _build_chart_story(lang: str = "cz") -> dict:
    history    = _state.get("history", [])
    prediction = _state.get("prediction", [])
    last_60    = history[-60:] if len(history) >= 60 else history
    hist_len   = len(last_60)
    habit_info = _state.get("habit_info")
    portfolio  = _state.get("portfolio_value", 0.0)

    def _czk(v: float) -> str:
        return f"{int(round(v)):,}".replace(",", " ") + " Kč"

    s = _T.get(lang, _T["cz"])["story"]   # story translation dict for this lang

    history_events: list[dict] = []

    # Card 1: income
    salary_hits = [(i, e) for i, e in enumerate(last_60) if e["income"] > 0]
    if salary_hits:
        idx, entry = salary_hits[-1]
        income_amt = int(entry["income"])
        bal_after  = int(entry["balance"])
        history_events.append({
            "chart_index": idx,
            "date":        entry["date"],
            "type":        "income",
            "label":       s["income_label"],
            "amount":      bal_after,
            "text":        s["income_text"].format(salary=_czk(income_amt), balance=_czk(bal_after)),
        })

    # Card 2: expense peak
    big_idx     = max(range(len(last_60)), key=lambda i: last_60[i]["expense"])
    big         = last_60[big_idx]
    expense_amt = int(big["expense"])
    bal_at_exp  = int(big["balance"])
    habit_t     = _t(lang, "habits", habit_info["scenario"]) if habit_info else {}
    habit_name  = (habit_t.get("name") if isinstance(habit_t, dict) else None) or \
                  (HABIT_META[habit_info["scenario"]]["name"] if habit_info else "")
    history_events.append({
        "chart_index": big_idx,
        "date":        big["date"],
        "type":        "expense",
        "label":       s["expense_label"],
        "amount":      expense_amt,
        "balance":     bal_at_exp,
        "text":        s["expense_text"].format(amount=_czk(expense_amt), habit=habit_name, balance=_czk(bal_at_exp)),
    })

    # Card 3: loan payments
    mortgage_payment  = _state.get("mortgage_payment", 18_000)
    mini_loan_payment = _state.get("mini_loan_payment", 4_000)
    loan_total        = mortgage_payment + mini_loan_payment
    loan_idx = next(
        (i for i in range(len(last_60) - 1, -1, -1) if last_60[i].get("mini_loan", 0) > 0), None,
    )
    loan_date = last_60[loan_idx]["date"] if loan_idx is not None else last_60[-1]["date"]
    history_events.append({
        "chart_index": loan_idx if loan_idx is not None else hist_len - 1,
        "date":        loan_date,
        "type":        "loans",
        "label":       s["loans_label"],
        "amount":      loan_total,
        "balance":     None,
        "text": (
            s["loans_text"].format(total=_czk(loan_total), mortgage=_czk(mortgage_payment), loan=_czk(mini_loan_payment))
            + "<br>" + s["smart_rate"]
        ),
    })

    prediction_events: list[dict] = []

    # Card 4: AI Autopilot
    agent_idx = next((i for i, e in enumerate(prediction) if e.get("is_agent")), None)
    if agent_idx is None:
        agent_idx = min(22, len(prediction) - 1)
    bal_after_agent = int(prediction[agent_idx]["balance"])
    etf_projected   = int(portfolio + SIMULATION_AMOUNT)
    monthly_savings = (habit_info["ev"] // 12) if habit_info else 0
    success_fee     = round(monthly_savings * 0.10)
    prediction_events.append({
        "chart_index":    hist_len + agent_idx,
        "date":           prediction[agent_idx]["date"],
        "type":           "agent",
        "label":          s["agent_label"],
        "amount":         SIMULATION_AMOUNT,
        "balance":        bal_after_agent,
        "etf_after":      etf_projected,
        "monthly_savings": monthly_savings,
        "success_fee":    success_fee,
        "text": (
            s["agent_text"].format(amount=_czk(SIMULATION_AMOUNT), etf=_czk(etf_projected))
            + "<br>" + s["success_fee"].format(savings=_czk(monthly_savings), fee=_czk(success_fee))
        ),
    })

    return {
        "history_events":    history_events,
        "prediction_events": prediction_events,
    }

def _build_monthly_chart(lang: str = "cz") -> dict:
    """Aggregate daily history/prediction into monthly grouped bar chart data."""
    history    = _state.get("history", [])
    prediction = _state.get("prediction", [])

    # Group history entries by (year, month)
    monthly: dict[tuple, list] = {}
    for entry in history:
        d   = date.fromisoformat(entry["date"])
        key = (d.year, d.month)
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(entry)

    sorted_keys = sorted(monthly.keys())
    last_6      = sorted_keys[-6:] if len(sorted_keys) >= 6 else sorted_keys

    labels:       list = []
    income_data:  list = []
    expense_data: list = []
    tooltips:     list = []

    def _sum_cat(entries: list, key: str) -> int:
        return sum(e.get(key, 0) for e in entries)

    for k in last_6:
        entries      = monthly[k]
        total_income = _sum_cat(entries, "income")
        total_exp    = _sum_cat(entries, "expense")
        labels.append(_t(lang, "months")[k[1] - 1])
        income_data.append(total_income)
        expense_data.append(total_exp)
        tooltips.append({
            "total_income":  total_income,
            "total_expense": total_exp,
            "cat_loans":     _sum_cat(entries, "cat_loans"),
            "cat_housing":   _sum_cat(entries, "cat_housing"),
            "cat_digital":   _sum_cat(entries, "cat_digital"),
            "cat_leisure":   _sum_cat(entries, "cat_leisure"),
            "cat_daily":     _sum_cat(entries, "cat_daily"),
        })

    # Aggregate prediction month
    pred_income  = _sum_cat(prediction, "income")
    pred_expense = _sum_cat(prediction, "expense")
    pred_agent   = next((SIMULATION_AMOUNT for e in prediction if e.get("is_agent")), 0)

    if prediction:
        month_counts: dict[int, int] = {}
        for e in prediction:
            m = date.fromisoformat(e["date"]).month
            month_counts[m] = month_counts.get(m, 0) + 1
        pred_label = _t(lang, "months")[max(month_counts, key=month_counts.get) - 1]
    else:
        pred_label = "Predikce"

    # ETF transfer is investment — exclude from displayed expense bar
    pred_exp_display = pred_expense - pred_agent

    labels.append(pred_label)
    income_data.append(pred_income)
    expense_data.append(pred_exp_display)
    tooltips.append({
        "total_income":  pred_income,
        "total_expense": pred_exp_display,
        "cat_loans":     _sum_cat(prediction, "cat_loans"),
        "cat_housing":   _sum_cat(prediction, "cat_housing"),
        "cat_digital":   _sum_cat(prediction, "cat_digital"),
        "cat_leisure":   _sum_cat(prediction, "cat_leisure"),
        "cat_daily":     _sum_cat(prediction, "cat_daily"),
        "ai_autopilot":  pred_agent,
        "is_prediction": True,
    })

    return {
        "months":                 labels,
        "income_data":            income_data,
        "expense_data":           expense_data,
        "tooltips":               tooltips,
        "prediction_month_index": len(labels) - 1,
    }


def _etf_5yr_gain(ev_annual: float, annual_return: float = 0.095) -> int:
    """Compound growth on redirected monthly savings over 5 years minus principal."""
    monthly_pmt = ev_annual / 12
    monthly_r   = annual_return / 12
    fv          = monthly_pmt * ((1 + monthly_r) ** 60 - 1) / monthly_r
    return int(fv - monthly_pmt * 60)


def _inflation_loss_5yr(ev_annual: float, inflation: float = 0.035) -> int:
    """Purchasing-power loss when EV savings sit at 0 % while inflation runs at `inflation` p.a."""
    total_nominal = ev_annual * 5
    real_value    = total_nominal / (1 + inflation) ** 5
    return int(total_nominal - real_value)


def _build_expense_donut(lang: str = "cz") -> dict:
    habit_info = _state.get("habit_info")

    total_expense = _state.get("monthly_expense", 47_000)

    # Start from base percentages; apply habit boost to the relevant category
    pcts      = {c["key"]: c["pct"] for c in _EXPENSE_CATS}
    alarm_key = None
    if habit_info:
        cat_key, boost = _HABIT_TO_CAT.get(habit_info["scenario"], (None, 0))
        if cat_key:
            alarm_key        = cat_key
            pcts[cat_key]   += boost
            # Absorb the boost from "daily" (catch-all); fall back to "housing"
            reduce_key       = "daily" if cat_key != "daily" else "housing"
            pcts[reduce_key] = max(5, pcts[reduce_key] - boost)
            diff             = 100 - sum(pcts.values())
            pcts[reduce_key] += diff  # normalise to exactly 100 %

    cats_t    = _t(lang, "cats")
    segments: list[dict] = []
    for cat in _EXPENSE_CATS:
        k        = cat["key"]
        pct      = pcts[k]
        is_alarm = k == alarm_key
        label    = cats_t.get(k, cat["label"]) if isinstance(cats_t, dict) else cat["label"]
        segments.append({
            "key":      k,
            "label":    label,
            "pct":      pct,
            "amount":   round(total_expense * pct / 100),
            "color":    "#FF9500" if is_alarm else cat["color"],
            "is_alarm": is_alarm,
            "offset":   15 if is_alarm else 0,
        })

    alarm_payload = None
    if alarm_key and habit_info:
        alarm_cat  = next(c for c in _EXPENSE_CATS if c["key"] == alarm_key)
        alarm_seg  = next(s for s in segments if s["key"] == alarm_key)
        excess_pct = max(0, alarm_seg["pct"] - alarm_cat["limit"])
        alarm_payload = {
            "category":          alarm_seg["label"],
            "current_pct":       alarm_seg["pct"],
            "excess_pct":        excess_pct,
            "monthly_savings":   habit_info["ev"] // 12,
            "etf_5yr_bonus":     _etf_5yr_gain(habit_info["ev"]),
            "inflation_loss_5yr": round(habit_info["ev"] * 0.12),
        }

    return {
        "total_expense": total_expense,
        "segments":      segments,
        "alarm":         alarm_payload,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state")
async def get_state(lang: str = "cz"):
    # safe_surplus = conservative monthly amount the agent can redirect to ETF
    monthly_salary  = _state.get("monthly_salary", 75_000)
    mortgage        = _state.get("mortgage_payment", 18_000)
    mini_loan       = _state.get("mini_loan_payment", 4_000)
    monthly_expense = _state.get("monthly_expense", 47_000)
    computed = monthly_salary - mortgage - mini_loan - monthly_expense
    safe_surplus = max(5_000, min(SIMULATION_AMOUNT, computed))
    return JSONResponse({
        "checking_balance": _state["checking_balance"],
        "savings_balance":  _state["savings_balance"],
        "portfolio_value":  _state["portfolio_value"],
        "portfolio_units":  _state["portfolio_units"],
        "etf_price":        _state["etf_price"],
        "insurance_due_days": INSURANCE_DUE_DAYS,
        "insurance_amount":   INSURANCE_AMOUNT,
        "agent_log":        _state["agent_log"][-20:],
        "detected_habits":  _state.get("detected_habits"),
        "chart":            _build_monthly_chart(lang),
        "chart_story":      _build_chart_story(lang),
        "expense_donut":    _build_expense_donut(lang),
        "safe_surplus":     int(safe_surplus),
    })


@app.post("/api/run-agent")
async def run_agent_endpoint(body: dict = Body(default={})):
    q1   = max(1, min(3, int(body.get("q1", 2))))
    q2   = max(1, min(3, int(body.get("q2", 2))))
    q3   = max(1, min(3, int(body.get("q3", 2))))
    lang = body.get("lang", "cz")

    risk_profile, score = _score_to_profile(q1, q2, q3)
    profile = RISK_PROFILES[risk_profile]

    _log(
        "mifid_result",
        f"Na základě MiFID II dotazníku (Skóre: {score}/9) byl zvolen profil "
        f"{profile['label']}. Alokuji prostředky. "
        f"Předpokládaný roční výnos portfolia: {profile['expected_value']:.1f} % p.a. "
        f"{_var_label_investment(profile['variance'])}. "
        f"(Model: E[X] = {profile['expected_value']:.1f}%, Var(X) = {profile['variance']:.4f})",
    )
    _log(
        "mifid_prescoring",
        "[MiFID Pre-scoring 📊] Načtení profilu klienta z core systému RB "
        "(Majetkové zázemí: OK, Roční obrat: OK, Čisté jmění ověřeno z historie transakcí).",
    )
    _log(
        "target_market",
        f"[iShares Target Market 🎯] Ověřena shoda produktu (iShares Core S&P 500 UCITS ETF) "
        f"s profilem investora '{profile['label']}'. "
        "Podmínky znalostí splněny automatickým scoringem investiční historie.",
    )

    horizon_years = _HORIZON_YEARS.get(q1, 5)
    summary       = run_agent(risk_profile)
    monthly_salary   = _state.get("monthly_salary", 75_000)
    mortgage         = _state.get("mortgage_payment", 18_000)
    mini_loan        = _state.get("mini_loan_payment", 4_000)
    monthly_expense  = _state.get("monthly_expense", 47_000)
    computed_surplus = monthly_salary - mortgage - mini_loan - monthly_expense
    safe_surplus     = max(5_000, min(SIMULATION_AMOUNT, computed_surplus))
    return JSONResponse({
        "summary":         summary,
        "agent_log":       _state["agent_log"][-20:],
        "risk_profile":    risk_profile,
        "score":           score,
        "safe_surplus":    int(safe_surplus),
        "detected_habits": _build_detected_habit(_state["habit_info"], lang) if _state.get("habit_info") else _state.get("detected_habits"),
        "chart_story":     _build_chart_story(lang),
        "projection":      _compute_investment_projection(
            SIMULATION_AMOUNT, horizon_years, risk_profile
        ),
        "horizon_q1":      q1,
    })


@app.post("/api/reset")
async def reset(body: dict = Body(default={})):
    lang = body.get("lang", "cz")

    # Seed from os.urandom to guarantee different values on every call,
    # regardless of clock resolution or server environment.
    seed = int.from_bytes(os.urandom(8), "big")
    rng  = random.Random(seed)
    initial_balance = rng.randint(45_000, 55_000)
    portfolio_val   = float(rng.randint(20_000, 35_000))
    etf_price       = 150.0

    _state["savings_balance"] = float(rng.randint(3_000, 8_000))
    _state["etf_price"]       = etf_price
    _state["portfolio_units"] = round(portfolio_val / etf_price, 2)
    _state["portfolio_value"] = round(_state["portfolio_units"] * etf_price, 2)
    _state["agent_log"]       = []

    _init_state(initial_balance, rng)   # picks new habit + generates fresh EV/Variance
    return JSONResponse({
        "ok":               True,
        "checking_balance": _state["checking_balance"],
        "savings_balance":  _state["savings_balance"],
        "portfolio_value":  _state["portfolio_value"],
        "portfolio_units":  _state["portfolio_units"],
        "etf_price":        _state["etf_price"],
        "detected_habits":  _state["detected_habits"],
        "chart":            _build_monthly_chart(lang),
        "chart_story":      _build_chart_story(lang),
        "expense_donut":    _build_expense_donut(lang),
    })
