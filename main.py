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
        "name":    "Gastro & Donáška – Lifestyle Creep",
        "context": "Výdaje za Wolt/Bolt Food a kavárny vzrostly meziměsíčně o 35 %.",
        "detail":  "meziměsíční nárůst výdajů za jídlo a donášku o 35 %",
        "ev_min":  7_000,  "ev_max":  10_000,
        "var_min": 0.0010, "var_max": 0.0015,
    },
    "subscription_trap": {
        "name":    "Subskripční peklo – Subscription Trap",
        "context": "Nahromadění 6+ drobných měsíčních plateb (Netflix, Spotify, SaaS, gym), které nejsou aktivně využívány.",
        "detail":  "6+ nevyužívaných měsíčních plateb (Netflix, Spotify, SaaS, gym)",
        "ev_min":  12_000, "ev_max":  16_000,
        "var_min": 0.0020, "var_max": 0.0030,
    },
    "weekend_micro": {
        "name":    "Víkendové mikro-transakce – Impulsive Spending",
        "context": "Vysoká frekvence drobných nákupů o víkendech (čerpací stanice, večerky, mikrotransakce).",
        "detail":  "vysoká frekvence drobných víkendových nákupů",
        "ev_min":  9_000,  "ev_max":  13_000,
        "var_min": 0.0015, "var_max": 0.0022,
    },
    "overpaying": {
        "name":    "Neloajalita vůči energiím – Overpaying Inertia",
        "context": "Fixní trvalé příkazy za pojištění a energie jsou o 20 % vyšší než průměr trhu.",
        "detail":  "fixní příkazy 20 % nad tržním průměrem (pojištění, energie)",
        "ev_min":  7_500,  "ev_max":  11_000,
        "var_min": 0.0007, "var_max": 0.0011,
    },
}

# ---------------------------------------------------------------------------
# Expense donut – category definitions for Revolut-style doughnut chart
# ---------------------------------------------------------------------------
_EXPENSE_CATS = [
    {"key": "housing",       "label": "Bydlení",    "pct": 40, "limit": 40, "color": "#3A4A5C"},
    {"key": "energy",        "label": "Energie",    "pct": 15, "limit": 15, "color": "#7B5EA7"},
    {"key": "gastro",        "label": "Gastro",     "pct": 15, "limit": 12, "color": "#FF6B35"},
    {"key": "subscriptions", "label": "Subskripce", "pct": 10, "limit":  8, "color": "#4A90D9"},
    {"key": "other",         "label": "Ostatní",    "pct": 20, "limit": 15, "color": "#6E6E73"},
]

# Maps each habit scenario → (alarmed_category_key, percentage_boost)
_HABIT_TO_CAT: dict[str, tuple[str, int]] = {
    "gastro_creep":      ("gastro",        12),
    "subscription_trap": ("subscriptions", 14),
    "weekend_micro":     ("other",         12),
    "overpaying":        ("energy",        10),
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


def _score_to_profile(q1: int, q2: int, q3: int) -> tuple[str, int]:
    score = q1 + q2 + q3
    if score <= 4:
        return "konzervativni", score
    if score <= 7:
        return "vyvazeny", score
    return "dynamicky", score


# ---------------------------------------------------------------------------
# Mock history – deterministic daily amounts (seed 42), variable start balance
# ---------------------------------------------------------------------------
def _build_history(initial_balance: int = 52_000) -> list[dict]:
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

        if d.day == 15:
            income += 65_000
        if d.day == 1:
            expense += 22_000
        if d.day == 5:
            expense += 4_500
        if d.day == 10:
            subscriptions = 1_500 if i <= 60 else 1_200
            expense += subscriptions

        daily = rng.randint(400, 950) if i <= 60 else rng.randint(300, 850)
        expense += daily

        if d.weekday() in (2, 5):   # Wed, Sat → gastro attribution
            gastro = round(daily * 0.55)

        if d.weekday() in (5, 6):   # Sat, Sun → impulse micro-transaction attribution
            weekend_micro = round(daily * 0.45)

        if i == 165:
            expense += INSURANCE_AMOUNT

        net = income - expense
        base_balance += net
        entries.append({
            "date": d.isoformat(),
            "income": income,
            "expense": expense,
            "net": net,
            "balance": base_balance,
            "gastro": gastro,
            "subscriptions": subscriptions,
            "weekend_micro": weekend_micro,
        })
    return entries


def _build_prediction(current_balance: float) -> list[dict]:
    entries = []
    balance = current_balance

    for i in range(1, 31):
        d = TODAY + timedelta(days=i)
        income = 0
        expense = 0

        if d.day == 15:
            income += 65_000
        if d.day == 1:
            expense += 22_000
        if d.day == 5:
            expense += 4_500
        if d.day == 10:
            expense += 1_500

        if i == INSURANCE_DUE_DAYS:
            expense += INSURANCE_AMOUNT

        expense += 675

        net = income - expense
        balance += net
        surplus = max(0.0, balance - 30_000)
        entries.append({
            "date": d.isoformat(),
            "income": income,
            "expense": expense,
            "net": net,
            "balance": balance,
            "surplus": surplus,
            "is_stress": i == INSURANCE_DUE_DAYS,
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
    history = _build_history(initial_balance)
    _state["history"] = history
    _state["checking_balance"] = history[-1]["balance"]
    _state["prediction"] = _build_prediction(_state["checking_balance"])
    if rng is None:
        rng = random.Random(int.from_bytes(os.urandom(8), "big"))
    habit = _pick_habit(rng)
    _state["habit_info"]       = habit
    _state["habit_scenario"]   = habit["scenario"]
    _state["detected_habits"]  = _build_detected_habit(habit)
    _state["monthly_expense"]  = rng.randint(30_000, 60_000)
    _state["monthly_salary"]   = rng.randint(55_000, 70_000)


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
        f"[Analýza chování ⚠️] Detekován zlozvyk: '{meta['name']}'. "
        f"{meta['context']} "
        f"Náš stochastický model předpovídá očekávanou roční ztrátu {ev:,} Kč. "
        f"Míra nejistoty (rozptyl) tohoto odhadu: Var = {variance:.4f}. "
        f"Agent doporučuje okamžité přesměrování ušetřené částky do {profile['etf_label']}.",
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
        "(po odečtení rezervy na blížící se pojistku).",
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
            f"Statistika portfolia: Očekávaná hodnota E[X] = {ev:.1f} % p.a., Rozptyl Var(X) = {var:.4f}."
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
            f"Statistika portfolia: Očekávaná hodnota E[X] = {ev:.1f} % p.a., Rozptyl Var(X) = {var:.4f}."
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
            f"Statistika portfolia: Očekávaná hodnota E[X] = {ev:.1f} % p.a., Rozptyl Var(X) = {var:.4f}."
        )

    _log("wealth_management", wm_msg)
    return f"Simulace dokončena. Profil: {profile['label']}. Přesunuto {amount:,} Kč."


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------
def run_agent(risk_profile: str = "vyvazeny") -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _run_fallback(risk_profile)

    prediction = _state["prediction"]
    safe_surplus = min(e["surplus"] for e in prediction[:INSURANCE_DUE_DAYS])
    stress_day = next((e for e in prediction if e.get("is_stress")), None)
    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["vyvazeny"])

    system_prompt = (
        f"Jsi autonomní finanční agent pro systém Closed-Loop Banking. "
        f"Rizikový profil uživatele (MiFID II): {profile['label']} "
        f"(ETF {round(profile['etf_pct']*100)} %, spoření {round(profile['savings_pct']*100)} %). "
        "Analyzuj cash flow, zohledni stress-test a proveď odpovídající přesuny pomocí nástrojů. "
        "Odpovídej česky, stručně a přesně."
    )

    user_msg = (
        f"Běžný účet: {_state['checking_balance']:,.0f} Kč\n"
        f"Spořicí účet: {_state['savings_balance']:,.0f} Kč\n"
        f"Portfolio (ETF): {_state['portfolio_value']:,.0f} Kč\n"
        f"Bezpečný přebytek před stress-testem: {safe_surplus:,.0f} Kč\n"
        f"Stress-test: za {INSURANCE_DUE_DAYS} dní pojistka {INSURANCE_AMOUNT:,} Kč "
        f"(prognóza zůstatku: {stress_day['balance']:,.0f} Kč)\n\n"
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
# Chart story – annotates key balance events with human-readable explanations
# ---------------------------------------------------------------------------
# Chart story – 4 dynamic milestone cards, all values from per-reset _state
# ---------------------------------------------------------------------------
def _build_chart_story() -> dict:
    history    = _state.get("history", [])
    prediction = _state.get("prediction", [])
    last_60    = history[-60:] if len(history) >= 60 else history
    hist_len   = len(last_60)
    habit_info = _state.get("habit_info")
    salary     = _state.get("monthly_salary", 65_000)
    checking   = _state.get("checking_balance", 0.0)

    def _czk(v: float) -> str:
        return f"{int(round(v)):,}".replace(",", " ") + " Kč"

    history_events: list[dict] = []

    # ── Card 1: V\u00fdplata ──────────────────────────────────────────────
    salary_hits = [(i, e) for i, e in enumerate(last_60) if e["income"] > 0]
    if salary_hits:
        idx, entry = salary_hits[-1]
        peak_balance = int(entry["balance"]) + (salary - 65_000)
        history_events.append({
            "chart_index": idx,
            "date":        entry["date"],
            "type":        "income",
            "label":       "V\u00fdplata",
            "amount":      salary,
            "text": (
                f"Na \u00fa\u010det dorazila pravideln\u00e1 m\u011bs\u00ed\u010dn\u00ed mzda {_czk(salary)}. "
                f"\u0160pi\u010dkov\u00fd z\u016fstatek po p\u0159ips\u00e1n\u00ed: {_czk(peak_balance)}."
            ),
        })

    # ── Card 2: V\u00fddajov\u00fd vrchol ───────────────────────────────────────────
    RENT, ENERGY  = 22_000, 4_500
    habit_monthly = (habit_info["ev"] // 12) if habit_info else 0
    habit_name    = HABIT_META[habit_info["scenario"]]["name"] if habit_info else "V\u00fddaje"
    expense_peak  = RENT + ENERGY + habit_monthly

    big_idx = max(range(len(last_60)), key=lambda i: last_60[i]["expense"])
    big     = last_60[big_idx]
    history_events.append({
        "chart_index": big_idx,
        "date":        big["date"],
        "type":        "expense",
        "label":       "V\u00fddajov\u00fd vrchol",
        "amount":      expense_peak,
        "text": (
            f"N\u00e1jemn\u00e9 {_czk(RENT)} + energie {_czk(ENERGY)} "
            f"+ zlozvyk [{habit_name}]: {_czk(habit_monthly)} m\u011bs\u00ed\u010dn\u011b. "
            f"V\u00fddajov\u00fd vrchol: {_czk(expense_peak)}."
        ),
    })

    prediction_events: list[dict] = []

    # ── Card 3: Stress-test ────────────────────────────────────────────────────
    balance_after    = checking - INSURANCE_AMOUNT
    stress_idx       = next((i for i, e in enumerate(prediction) if e.get("is_stress")), None)
    stress_date      = (
        prediction[stress_idx]["date"] if stress_idx is not None
        else (TODAY + timedelta(days=INSURANCE_DUE_DAYS)).isoformat()
    )
    chart_stress_idx = hist_len + (stress_idx if stress_idx is not None else INSURANCE_DUE_DAYS - 1)
    prediction_events.append({
        "chart_index": chart_stress_idx,
        "date":        stress_date,
        "type":        "stress",
        "label":       "Stress-test",
        "amount":      INSURANCE_AMOUNT,
        "text": (
            f"Predikovan\u00fd stress-test: ro\u010dn\u00ed pojistka {_czk(INSURANCE_AMOUNT)}. "
            f"Predikovan\u00fd z\u016fstatek po platb\u011b: {_czk(balance_after)}."
        ),
    })

    # ── Card 4: AI Autopilot ──────────────────────────────────────────────────
    safe_surplus   = max(0.0, balance_after - 30_000)
    agent_pred_idx = min(INSURANCE_DUE_DAYS + 2, len(prediction) - 1)
    agent_date     = (
        prediction[agent_pred_idx]["date"] if prediction
        else (TODAY + timedelta(days=INSURANCE_DUE_DAYS + 3)).isoformat()
    )
    prediction_events.append({
        "chart_index": hist_len + agent_pred_idx,
        "date":        agent_date,
        "type":        "agent",
        "label":       "AI Autopilot",
        "amount":      SIMULATION_AMOUNT,
        "text": (
            f"AI Autopilot detekoval bezpe\u010dn\u00fd p\u0159ebytek {_czk(safe_surplus)} "
            f"po splatnosti pojistky. Odkl\u00e1n\u00ed {_czk(SIMULATION_AMOUNT)} do ETF portfolia \u2013 "
            f"sni\u017euje z\u016fstatek b\u011b\u017en\u00e9ho \u00fa\u010dtu, ale buduje r\u016fstovou vrstvu majetku."
        ),
    })

    return {
        "history_events":    history_events,
        "prediction_events": prediction_events,
    }
def _etf_5yr_gain(ev_annual: float, annual_return: float = 0.095) -> int:
    """Compound growth on redirected monthly savings over 5 years minus principal."""
    monthly_pmt = ev_annual / 12
    monthly_r   = annual_return / 12
    fv          = monthly_pmt * ((1 + monthly_r) ** 60 - 1) / monthly_r
    return int(fv - monthly_pmt * 60)


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
            reduce_key       = "other" if cat_key != "other" else "housing"
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
            "category":        alarm_seg["label"],
            "current_pct":     alarm_seg["pct"],
            "excess_pct":      excess_pct,
            "monthly_savings": habit_info["ev"] // 12,
            "etf_5yr_bonus":   _etf_5yr_gain(habit_info["ev"]),
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
    history = _state["history"]
    prediction = _state["prediction"]
    chart_labels = [e["date"] for e in history[-60:]] + [e["date"] for e in prediction]

    return JSONResponse({
        "checking_balance": _state["checking_balance"],
        "savings_balance": _state["savings_balance"],
        "portfolio_value": _state["portfolio_value"],
        "portfolio_units": _state["portfolio_units"],
        "etf_price": _state["etf_price"],
        "insurance_due_days": INSURANCE_DUE_DAYS,
        "insurance_amount": INSURANCE_AMOUNT,
        "agent_log": _state["agent_log"][-20:],
        "detected_habits": _state.get("detected_habits"),  # None until first agent run
        "chart": {
            "labels": chart_labels,
            "history_balance": [e["balance"] for e in history[-60:]],
            "predicted_balance": [e["balance"] for e in prediction],
            "stress_index": len(history[-60:]) + INSURANCE_DUE_DAYS - 1,
        },
        "chart_story":    _build_chart_story(),
        "expense_donut":  _build_expense_donut(),
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
        f"Parametry distribuce portfolia: "
        f"Očekávaná hodnota = {profile['expected_value']:.1f} % p.a., "
        f"Rozptyl = {profile['variance']:.4f}.",
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
