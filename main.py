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
    "detected_habits": None,   # None = not yet analyzed
    "habit_scenario": "gastro_creep",
}


def _init_state(initial_balance: int = 52_000) -> None:
    history = _build_history(initial_balance)
    _state["history"] = history
    _state["checking_balance"] = history[-1]["balance"]
    _state["prediction"] = _build_prediction(_state["checking_balance"])


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
# 4-habit spending behaviour detection with EV / Variance per habit
# ---------------------------------------------------------------------------
def _detect_spending_habits(risk_profile: str) -> list[dict]:
    """Detects one of 4 financial bad habits and logs EV/Variance warning."""
    history = _state["history"]
    if len(history) < 60:
        _state["detected_habits"] = []
        return []

    scenario = _state.get("habit_scenario", "gastro_creep")
    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["vyvazeny"])
    detected: list[dict] = []

    if scenario == "gastro_creep":
        recent_gastro = sum(e.get("gastro", 0) for e in history[-60:])
        prior_gastro = (
            sum(e.get("gastro", 0) for e in history[-120:-60])
            if len(history) >= 120 else round(recent_gastro * 0.74)
        )
        growth_pct = round((recent_gastro / prior_gastro - 1) * 100) if prior_gastro > 0 else 35
        monthly_excess = max(round((recent_gastro - prior_gastro) / 2), 400)
        ev_loss = monthly_excess * 12
        variance = 0.0045
        _log(
            "spending_warning",
            f"[Analýza chování ⚠️] Detekován zlozvyk: 'Gastro & Donáška – Lifestyle Creep' "
            f"(výdaje za jídlo a donášku vzrostly o {growth_pct} % meziměsíčně). "
            f"Pokud nezasáhnete, Expected Value finanční ztráty za 12 měsíců je "
            f"{ev_loss:,} Kč při Variance = {variance}. "
            f"Agent doporučuje nastavení měsíčního limitu a přesměrování úspory do {profile['etf_label']}.",
        )
        detected.append({
            "key": "gastro_creep",
            "name": "Gastro & Donáška – Lifestyle Creep",
            "detail": f"výdaje vzrostly o {growth_pct} % MoM",
            "ev_loss": ev_loss,
            "variance": variance,
        })

    elif scenario == "subscription_trap":
        active_services = 6
        monthly_subs = max(round(sum(e.get("subscriptions", 0) for e in history[-30:])), 1_200)
        ev_loss = monthly_subs * 12
        variance = 0.0025
        _log(
            "spending_warning",
            f"[Analýza chování ⚠️] Detekován zlozvyk: 'Subskripční peklo – Subscription Trap' "
            f"({active_services} aktivních předplatných služeb). "
            f"Pokud nezasáhnete, Expected Value finanční ztráty za 12 měsíců je "
            f"{ev_loss:,} Kč při Variance = {variance}. "
            f"Agent doporučuje konsolidaci a okamžité přesměrování ušetřené částky do {profile['etf_label']}.",
        )
        detected.append({
            "key": "subscription_trap",
            "name": "Subskripční peklo – Subscription Trap",
            "detail": f"{active_services} aktivních předplatných služeb",
            "ev_loss": ev_loss,
            "variance": variance,
        })

    elif scenario == "weekend_micro":
        weekend_entries = [
            e for e in history[-60:]
            if date.fromisoformat(e["date"]).weekday() >= 5
        ]
        avg_micro = (
            sum(e.get("weekend_micro", 0) for e in weekend_entries) / len(weekend_entries)
            if weekend_entries else 550
        )
        monthly_excess = round(avg_micro * 8)
        ev_loss = max(monthly_excess * 12, 3_600)
        variance = 0.0180
        _log(
            "spending_warning",
            f"[Analýza chování ⚠️] Detekován zlozvyk: 'Víkendové mikro-transakce – Impulsive Spending' "
            f"(průměrně {round(avg_micro):,} Kč/víkendový den v impulzivních nákupech). "
            f"Pokud nezasáhnete, Expected Value finanční ztráty za 12 měsíců je "
            f"{ev_loss:,} Kč při Variance = {variance}. "
            f"Agent doporučuje nastavení víkendového limitu a automatický převod úspory do {profile['etf_label']}.",
        )
        detected.append({
            "key": "weekend_micro",
            "name": "Víkendové mikro-transakce – Impulsive Spending",
            "detail": f"průměrně {round(avg_micro):,} Kč na víkendový den",
            "ev_loss": ev_loss,
            "variance": variance,
        })

    elif scenario == "overpaying":
        monthly_overpay = round(INSURANCE_AMOUNT * 0.20 / 12)
        ev_loss = monthly_overpay * 12
        variance = 0.0008
        _log(
            "spending_warning",
            f"[Analýza chování ⚠️] Detekován zlozvyk: 'Neloajalita – Overpaying Inertia' "
            f"(fixní příkazy pojistka/energie jsou o 20 % nad tržním průměrem, "
            f"přeplatek ~{monthly_overpay:,} Kč/měsíc). "
            f"Pokud nezasáhnete, Expected Value finanční ztráty za 12 měsíců je "
            f"{ev_loss:,} Kč při Variance = {variance}. "
            f"Agent doporučuje okamžité srovnání trhu a přesměrování úspory do {profile['etf_label']}.",
        )
        detected.append({
            "key": "overpaying",
            "name": "Neloajalita – Overpaying Inertia",
            "detail": f"přeplatek ~{monthly_overpay:,} Kč/měsíc oproti tržnímu průměru",
            "ev_loss": ev_loss,
            "variance": variance,
        })

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
            f"[Wealth Management 📈] Na základě vašeho dlouhého investičního horizontu (7+ let) "
            f"a vysoké tolerance k riziku jsem ušetřených {amount:,} Kč zainvestoval následovně: "
            f"{sp500_amt:,} Kč (70 %) směřuje do fondu iShares Core S&P 500 UCITS ETF, "
            f"který kopíruje 500 největších amerických firem (Apple, Microsoft, Nvidia) "
            f"pro maximální růstový potenciál. "
            f"Zbylých {msci_amt:,} Kč (30 %) jsem alokoval do Amundi MSCI World ETF "
            f"pro globální diverzifikaci napříč Evropou a Japonskem. "
            f"Distribuce výnosů vašeho nového portfolia: "
            f"Očekávaná hodnota E[X] = {ev:.1f} % p.a., Rozptyl Var(X) = {var:.4f}."
        )
    elif risk_profile == "vyvazeny":
        eq_amt   = round(amount * 0.50)
        bond_amt = amount - eq_amt
        wm_msg = (
            f"[Wealth Management ⚖️] Vzhledem k vašemu střednímu horizontu (3–7 let) "
            f"volím vyvážený přístup. Částku {amount:,} Kč rozděluji na "
            f"{eq_amt:,} Kč (50 %) do Vanguard FTSE All-World ETF (akciový růst) "
            f"a {bond_amt:,} Kč (50 %) do iShares Core Global Aggregate Bond ETF "
            f"(stabilní dluhopisy, které tlumí případné propady trhu). "
            f"Distribuce výnosů: Očekávaná hodnota E[X] = {ev:.1f} % p.a., "
            f"Rozptyl Var(X) = {var:.4f}."
        )
    else:  # konzervativni
        bond_amt   = round(amount * 0.80)
        liquid_amt = amount - bond_amt
        wm_msg = (
            f"[Wealth Management 🛡️] Jelikož preferujete maximální bezpečí a ochranu před inflací, "
            f"částku {amount:,} Kč jsem alokoval ze 80 % ({bond_amt:,} Kč) do "
            f"iShares EUR Ultrashort Bond ETF "
            f"(vysoce bezpečné krátkodobé evropské dluhopisy) "
            f"a 20 % ({liquid_amt:,} Kč) jsem přesunul na váš Spořicí účet Raiffeisenbank "
            f"pro okamžitou likviditu. "
            f"Distribuce výnosů: Očekávaná hodnota E[X] = {ev:.1f} % p.a., "
            f"Rozptyl Var(X) = {var:.4f}."
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
    rng = random.Random()           # entropy-seeded → different values each call
    initial_balance = rng.randint(45_000, 55_000)
    portfolio_val   = float(rng.randint(20_000, 35_000))
    etf_price       = 150.0

    _state["savings_balance"]  = float(rng.randint(3_000, 8_000))
    _state["etf_price"]        = etf_price
    _state["portfolio_units"]  = round(portfolio_val / etf_price, 2)
    _state["portfolio_value"]  = round(_state["portfolio_units"] * etf_price, 2)
    _state["agent_log"]        = []
    _state["detected_habits"]  = None
    _state["habit_scenario"]   = rng.choice(HABIT_SCENARIOS)

    _init_state(initial_balance)
    return JSONResponse({
        "ok": True,
        "checking_balance": _state["checking_balance"],
        "savings_balance":  _state["savings_balance"],
        "portfolio_value":  _state["portfolio_value"],
        "portfolio_units":  _state["portfolio_units"],
        "etf_price":        _state["etf_price"],
    })
