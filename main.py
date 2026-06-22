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
        "etf_label": "ETF Peněžní trh",
        "savings_label": "Dluhopisy",
        "annual_return": 0.04,
        "expected_value": 4.0,
        "variance": 0.0015,
    },
    "vyvazeny": {
        "label": "Vyvážený",
        "etf_pct": 0.50,
        "savings_pct": 0.50,
        "etf_label": "Akciové globální ETF",
        "savings_label": "Dluhopisy",
        "annual_return": 0.065,
        "expected_value": 6.5,
        "variance": 0.0080,
    },
    "dynamicky": {
        "label": "Dynamický",
        "etf_pct": 0.80,
        "savings_pct": 0.20,
        "etf_label": "Akciové ETF S&P500 / MSCI World + Krypto/Tech ETF",
        "savings_label": "Likvidní rezerva",
        "annual_return": 0.095,
        "expected_value": 9.5,
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


def _var_label_spending(var: float) -> str:
    """Human-readable risk label for spending-habit context."""
    if var < 0.0012:
        return "nízká (stabilní, předvídatelný vzorec)"
    if var < 0.0022:
        return "mírná (občasné výkyvy)"
    return "vyšší (proměnlivý výdajový vzorec)"


def _score_to_profile(q1: int, q2: int, q3: int) -> tuple[str, int]:
    score = q1 + q2 + q3
    if score <= 4:
        return "konzervativni", score
    if score <= 7:
        return "vyvazeny", score
    return "dynamicky", score


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
    last60: list[dict], habit_scenario: str | None, rng: random.Random
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
) -> list[dict]:
    rng = random.Random(42)
    entries = []
    base_balance = initial_balance

    for i in range(180, 0, -1):
        d = TODAY - timedelta(days=i)
        income = 0
        expense = 0
        gastro = 0
        subscriptions = 0
        weekend_micro = 0
        mortgage = 0
        mini_loan = 0

        if d.day == 15:
            income += 65_000
        if d.day == 1:
            expense += 22_000
        if d.day == 2:
            mortgage = mortgage_payment
            expense += mortgage
        if d.day == 5:
            expense += 4_500
        if d.day == 10:
            subscriptions = 1_500 if i <= 60 else 1_200
            expense += subscriptions
        if d.day == 18:
            mini_loan = mini_loan_payment
            expense += mini_loan

        daily = rng.randint(400, 950) if i <= 60 else rng.randint(300, 850)
        expense += daily

        if d.weekday() in (2, 5):
            gastro = round(daily * 0.55)
        if d.weekday() in (5, 6):
            weekend_micro = round(daily * 0.45)

        net = income - expense
        base_balance += net
        entries.append({
            "date":          d.isoformat(),
            "income":        income,
            "expense":       expense,
            "net":           net,
            "balance":       base_balance,
            "gastro":        gastro,
            "subscriptions": subscriptions,
            "weekend_micro": weekend_micro,
            "mortgage":      mortgage,
            "mini_loan":     mini_loan,
            # Clean spending categories for monthly chart aggregation
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
            income += 65_000
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


def _build_detected_habit(habit: dict) -> list[dict]:
    """Build the detected_habits payload from a habit_info dict (no log entry)."""
    meta = HABIT_META[habit["scenario"]]
    return [{
        "key":      habit["scenario"],
        "name":     meta["name"],
        "detail":   meta["detail"],
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

    history = _build_history(initial_balance, mortgage_payment, mini_loan_payment)
    _state["history"] = history
    _state["checking_balance"] = history[-1]["balance"]
    _state["prediction"] = _build_prediction(
        _state["checking_balance"], mortgage_payment, mini_loan_payment
    )
    habit = _pick_habit(rng)
    _state["habit_info"]       = habit
    _state["habit_scenario"]   = habit["scenario"]
    _state["detected_habits"]  = _build_detected_habit(habit)
    _state["monthly_expense"]  = rng.randint(30_000, 60_000)
    _state["monthly_salary"]   = rng.randint(55_000, 70_000)
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
def _build_chart_story() -> dict:
    history    = _state.get("history", [])
    prediction = _state.get("prediction", [])
    last_60    = history[-60:] if len(history) >= 60 else history
    hist_len   = len(last_60)
    habit_info = _state.get("habit_info")
    portfolio  = _state.get("portfolio_value", 0.0)

    def _czk(v: float) -> str:
        return f"{int(round(v)):,}".replace(",", " ") + " Kč"

    history_events: list[dict] = []

    # Card 1: Výplata – balance read directly from chart array, no adjustments
    salary_hits = [(i, e) for i, e in enumerate(last_60) if e["income"] > 0]
    if salary_hits:
        idx, entry = salary_hits[-1]
        income_amt = int(entry["income"])   # 65 000 from history
        bal_after  = int(entry["balance"])  # exact value the chart plots
        history_events.append({
            "chart_index": idx,
            "date":        entry["date"],
            "type":        "income",
            "label":       "Výplata",
            "amount":      bal_after,        # green number = resulting balance
            "text": (
                f"Na účet dorazila pravidelná mzda {_czk(income_amt)}. "
                f"Zůstatek po připsání: {_czk(bal_after)}."
            ),
        })

    # Card 2: Výdajový vrchol – amounts come from the entry, not from formulas
    big_idx     = max(range(len(last_60)), key=lambda i: last_60[i]["expense"])
    big         = last_60[big_idx]
    expense_amt = int(big["expense"])       # exact expense the chart reflects
    bal_at_exp  = int(big["balance"])       # exact balance the chart shows
    habit_name  = HABIT_META[habit_info["scenario"]]["name"] if habit_info else "Výdaje"
    history_events.append({
        "chart_index": big_idx,
        "date":        big["date"],
        "type":        "expense",
        "label":       "Výdajový vrchol",
        "amount":      expense_amt,          # red number = total expense
        "balance":     bal_at_exp,           # displayed in footer
        "text": (
            f"Výdajový vrchol {_czk(expense_amt)}: nájemné, energie a neefektivní výdaj [{habit_name}]. "
            f"Zůstatek po odchodu plateb: {_czk(bal_at_exp)}."
        ),
    })

    # Card 3: Pravidelné závazky – most-recent month's loan payments from history
    mortgage_payment  = _state.get("mortgage_payment", 18_000)
    mini_loan_payment = _state.get("mini_loan_payment", 4_000)
    loan_total        = mortgage_payment + mini_loan_payment

    loan_idx = next(
        (i for i in range(len(last_60) - 1, -1, -1) if last_60[i].get("mini_loan", 0) > 0),
        None,
    )
    loan_date = last_60[loan_idx]["date"] if loan_idx is not None else last_60[-1]["date"]
    history_events.append({
        "chart_index": loan_idx if loan_idx is not None else hist_len - 1,
        "date":        loan_date,
        "type":        "loans",
        "label":       "Pravidelné závazky 🏠",
        "amount":      loan_total,
        "balance":     None,
        "text": (
            f"Odešlo na splátky úvěrů RB: {_czk(loan_total)}. "
            f"Splátka hypotéky: {_czk(mortgage_payment)}, "
            f"Minutová půjčka RB: {_czk(mini_loan_payment)}."
        ),
    })

    prediction_events: list[dict] = []

    # Card 4: AI Autopilot – ETF transfer + habit spending reduction
    agent_idx = next((i for i, e in enumerate(prediction) if e.get("is_agent")), None)
    if agent_idx is None:
        agent_idx = min(22, len(prediction) - 1)
    bal_after_agent = int(prediction[agent_idx]["balance"])
    etf_projected   = int(portfolio + SIMULATION_AMOUNT)
    prediction_events.append({
        "chart_index": hist_len + agent_idx,
        "date":        prediction[agent_idx]["date"],
        "type":        "agent",
        "label":       "AI Autopilot",
        "amount":      SIMULATION_AMOUNT,
        "balance":     bal_after_agent,
        "etf_after":   etf_projected,
        "text": (
            f"AI Autopilot detekoval skrytý únik peněz a zkrotil výdaje na Gastro & Subskripce. "
            f"Variabilní výdaje sníženy o ~33 %. Přebytek {_czk(SIMULATION_AMOUNT)} přesměrován do ETF portfolia. "
            f"Projekce ETF po převodu: {_czk(etf_projected)}."
        ),
    })

    return {
        "history_events":    history_events,
        "prediction_events": prediction_events,
    }
_CS_MONTHS = [
    "Leden", "Únor", "Březen", "Duben", "Květen", "Červen",
    "Červenec", "Srpen", "Září", "Říjen", "Listopad", "Prosinec",
]


def _month_name_cs(month: int) -> str:
    return _CS_MONTHS[month - 1]


def _build_monthly_chart() -> dict:
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
        labels.append(_month_name_cs(k[1]))
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
        pred_label = _month_name_cs(max(month_counts, key=month_counts.get))
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


def _build_expense_donut() -> dict:
    history    = _state.get("history", [])
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

    segments: list[dict] = []
    for cat in _EXPENSE_CATS:
        k        = cat["key"]
        pct      = pcts[k]
        is_alarm = k == alarm_key
        segments.append({
            "key":      k,
            "label":    cat["label"],
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
async def get_state():
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
        "chart":            _build_monthly_chart(),
        "chart_story":      _build_chart_story(),
        "expense_donut":    _build_expense_donut(),
    })


@app.post("/api/run-agent")
async def run_agent_endpoint(body: dict = Body(default={})):
    q1 = max(1, min(3, int(body.get("q1", 2))))
    q2 = max(1, min(3, int(body.get("q2", 2))))
    q3 = max(1, min(3, int(body.get("q3", 2))))

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

    summary = run_agent(risk_profile)
    return JSONResponse({
        "summary": summary,
        "agent_log": _state["agent_log"][-20:],
        "risk_profile": risk_profile,
        "score": score,
        "detected_habits": _state.get("detected_habits"),
    })


@app.post("/api/reset")
async def reset():
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
        "chart_story":      _build_chart_story(),    # story panel updates immediately
        "expense_donut":    _build_expense_donut(),  # donut updates immediately
    })
