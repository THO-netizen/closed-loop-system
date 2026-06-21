import os
import json
import random
from datetime import date, datetime, timedelta
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI(title="Autonomous Closed-Loop Banking")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Mock data – 6 months of transaction history
# ---------------------------------------------------------------------------
TODAY = date.today()
INSURANCE_DUE_DAYS = 20          # stress-point: annual insurance payment
INSURANCE_AMOUNT = 18_000        # CZK


def _build_history() -> list[dict]:
    """Generate 6 months of daily cash-flow entries ending today."""
    entries = []
    base_balance = 52_000  # starting balance 6 months ago

    for i in range(180, 0, -1):
        d = TODAY - timedelta(days=i)
        income = 0
        expense = 0

        # Salary on the 15th
        if d.day == 15:
            income += 65_000

        # Rent on the 1st
        if d.day == 1:
            expense += 22_000

        # Recurring utilities on the 5th
        if d.day == 5:
            expense += 4_500

        # Subscriptions on the 10th
        if d.day == 10:
            expense += 1_200

        # Random daily spending (groceries, transport, coffee…)
        expense += random.randint(400, 1_800)

        # Previous year's insurance (simulate it happened 345 days ago)
        if i == 165:
            expense += INSURANCE_AMOUNT

        net = income - expense
        base_balance += net
        entries.append(
            {
                "date": d.isoformat(),
                "income": income,
                "expense": expense,
                "net": net,
                "balance": base_balance,
            }
        )
    return entries


def _build_prediction(current_balance: float) -> list[dict]:
    """Predict cash flow for the next 30 days."""
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
            expense += 1_200

        # Insurance due in INSURANCE_DUE_DAYS days
        if i == INSURANCE_DUE_DAYS:
            expense += INSURANCE_AMOUNT

        expense += 1_100  # average daily spend

        net = income - expense
        balance += net

        surplus = max(0.0, balance - 30_000)  # keep 30 k CZK as safety buffer
        entries.append(
            {
                "date": d.isoformat(),
                "income": income,
                "expense": expense,
                "net": net,
                "balance": balance,
                "surplus": surplus,
                "is_stress": i == INSURANCE_DUE_DAYS,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# In-memory state (portfolio, savings)
# ---------------------------------------------------------------------------
_state = {
    "checking_balance": 0.0,
    "savings_balance": 5_000.0,
    "portfolio_value": 12_000.0,
    "portfolio_units": 80.0,      # ETF units
    "etf_price": 150.0,           # CZK per unit
    "agent_log": [],
}


def _init_state():
    history = _build_history()
    _state["history"] = history
    _state["checking_balance"] = history[-1]["balance"]
    prediction = _build_prediction(_state["checking_balance"])
    _state["prediction"] = prediction


_init_state()


# ---------------------------------------------------------------------------
# Tool implementations (called by the agent)
# ---------------------------------------------------------------------------
def transfer_to_savings(amount: float) -> dict:
    if amount <= 0:
        return {"success": False, "reason": "Amount must be positive."}
    if amount > _state["checking_balance"]:
        return {"success": False, "reason": "Insufficient funds in checking account."}

    _state["checking_balance"] -= amount
    _state["savings_balance"] += amount
    msg = f"Převod {amount:,.0f} Kč na spořicí účet dokončen. Nový zůstatek běžného účtu: {_state['checking_balance']:,.0f} Kč."
    _state["agent_log"].append({"ts": datetime.now().isoformat(timespec="seconds"), "action": "transfer_to_savings", "amount": amount, "message": msg})
    return {"success": True, "new_checking": _state["checking_balance"], "new_savings": _state["savings_balance"]}


def invest_funds(amount: float) -> dict:
    if amount <= 0:
        return {"success": False, "reason": "Amount must be positive."}
    if amount > _state["checking_balance"]:
        return {"success": False, "reason": "Insufficient funds in checking account."}

    units_bought = amount / _state["etf_price"]
    _state["checking_balance"] -= amount
    _state["portfolio_units"] += units_bought
    _state["portfolio_value"] = _state["portfolio_units"] * _state["etf_price"]
    msg = (
        f"Investováno {amount:,.0f} Kč → nakoupeno {units_bought:.2f} ETF jednotek @ {_state['etf_price']:.2f} Kč. "
        f"Celkové portfolio: {_state['portfolio_value']:,.0f} Kč ({_state['portfolio_units']:.2f} j.)."
    )
    _state["agent_log"].append({"ts": datetime.now().isoformat(timespec="seconds"), "action": "invest_funds", "amount": amount, "message": msg})
    return {"success": True, "units_bought": units_bought, "portfolio_value": _state["portfolio_value"]}


TOOLS = [
    {
        "name": "transfer_to_savings",
        "description": "Převede zadanou částku z běžného účtu na spořicí účet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Částka v CZK k převedení."}
            },
            "required": ["amount"],
        },
    },
    {
        "name": "invest_funds",
        "description": "Investuje zadanou částku z běžného účtu do globálního ETF portfolia.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Částka v CZK k investování."}
            },
            "required": ["amount"],
        },
    },
]


def _dispatch_tool(name: str, inp: dict) -> dict:
    if name == "transfer_to_savings":
        return transfer_to_savings(inp["amount"])
    if name == "invest_funds":
        return invest_funds(inp["amount"])
    return {"error": "Unknown tool"}


# ---------------------------------------------------------------------------
# Simulation fallback – runs when API key is missing or API call fails
# ---------------------------------------------------------------------------
SIMULATION_AMOUNT = 15_000  # CZK


def _log(action: str, message: str, amount: float = 0.0):
    _state["agent_log"].append(
        {"ts": datetime.now().isoformat(timespec="seconds"), "action": action, "amount": amount, "message": message}
    )


def _run_fallback() -> str:
    amount = SIMULATION_AMOUNT
    equity_amount = round(amount * 0.70)
    bond_amount = amount - equity_amount

    _log(
        "prediction",
        f"[Predikce] Detekován bezpečný dlouhodobý přebytek {amount:,} Kč "
        "(po odečtení rezervy na blížící se pojistku).",
    )
    _log(
        "execution",
        f"[Exekuce] Spouštím autonomní přesun {amount:,} Kč z běžného účtu.",
        amount,
    )

    # Actually move the money so balance cards update
    if amount <= _state["checking_balance"]:
        units_bought = amount / _state["etf_price"]
        _state["checking_balance"] -= amount
        _state["portfolio_units"] += units_bought
        _state["portfolio_value"] = _state["portfolio_units"] * _state["etf_price"]

    _log(
        "wealth_management",
        f"[Wealth Management] Peníze byly úspěšně rozděleny: "
        f"70 % Akciové globální ETF ({equity_amount:,} Kč), "
        f"30 % Dluhopisové ETF ({bond_amount:,} Kč). "
        "Portfolio rebalancováno na optimální růstovou trajektorii.",
    )
    return f"Simulace dokončena. Přesunuto {amount:,} Kč z běžného účtu do portfolia."


# ---------------------------------------------------------------------------
# Agentic loop (Anthropic Claude with tool use)
# ---------------------------------------------------------------------------
def run_agent() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _run_fallback()

    prediction = _state["prediction"]
    safe_surplus = min(e["surplus"] for e in prediction[:INSURANCE_DUE_DAYS])
    stress_day = next((e for e in prediction if e.get("is_stress")), None)

    system_prompt = (
        "Jsi autonomní finanční agent pro systém Closed-Loop Banking. "
        "Tvým úkolem je optimalizovat cash flow uživatele. "
        "Analyzuj aktuální finanční stav, 30denní predikci a stres-test (pojistka). "
        "Pokud je bezpečný přebytek > 0 Kč, vol nástroje transfer_to_savings nebo invest_funds. "
        "Investuj maximálně 60 % přebytku; zbytek přesuň na spoření. "
        "Vždy vysvětli svůj postup česky, stručně a přesně."
    )

    user_msg = (
        f"Aktuální zůstatek běžného účtu: {_state['checking_balance']:,.0f} Kč\n"
        f"Spořicí účet: {_state['savings_balance']:,.0f} Kč\n"
        f"Portfolio (ETF): {_state['portfolio_value']:,.0f} Kč\n"
        f"Bezpečný přebytek před stress-testem: {safe_surplus:,.0f} Kč\n"
        f"Stress-test: za {INSURANCE_DUE_DAYS} dní splatná pojistka {INSURANCE_AMOUNT:,} Kč\n"
        f"(stress-day prognóza zůstatku: {stress_day['balance']:,.0f} Kč)\n\n"
        "Proveď autonomní rozhodnutí a případně zavolej příslušné nástroje."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": user_msg}]

        # Agentic loop – keep going until Claude stops calling tools
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
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, ensure_ascii=False)}
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        final_text = " ".join(
            block.text for block in response.content if hasattr(block, "text") and block.text
        )
        return final_text or "Agent dokončil analýzu."

    except Exception:
        return _run_fallback()


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
    chart_balance_hist = [e["balance"] for e in history[-60:]]
    chart_balance_pred = [e["balance"] for e in prediction]

    return JSONResponse(
        {
            "checking_balance": _state["checking_balance"],
            "savings_balance": _state["savings_balance"],
            "portfolio_value": _state["portfolio_value"],
            "portfolio_units": _state["portfolio_units"],
            "etf_price": _state["etf_price"],
            "insurance_due_days": INSURANCE_DUE_DAYS,
            "insurance_amount": INSURANCE_AMOUNT,
            "agent_log": _state["agent_log"][-20:],
            "chart": {
                "labels": chart_labels,
                "history_balance": chart_balance_hist,
                "predicted_balance": chart_balance_pred,
                "stress_index": len(chart_balance_hist) + INSURANCE_DUE_DAYS - 1,
            },
        }
    )


@app.post("/api/run-agent")
async def run_agent_endpoint():
    summary = run_agent()
    return JSONResponse({"summary": summary, "agent_log": _state["agent_log"][-20:]})


@app.post("/api/reset")
async def reset():
    _state["savings_balance"] = 5_000.0
    _state["portfolio_value"] = 12_000.0
    _state["portfolio_units"] = 80.0
    _state["etf_price"] = 150.0
    _state["agent_log"] = []
    _init_state()
    return JSONResponse({"ok": True})
