import json
from openai import OpenAI

from rag import (
    create_mysql_engine,
    get_vehicle_analysis,
)

engine = create_mysql_engine()
client = OpenAI()


# ============================================================
# TOOL DEFINITION
# ============================================================

TOOLS = [
    {
        "type": "function",

        "name": "get_vehicle_analysis",

        "description": (
            "Retrieve FleetCare prediction information for a "
            "specific vehicle. Use this tool when the user asks "
            "about a vehicle's current state, risk, prediction "
            "history, deterioration, maintenance recommendation, "
            "or why the vehicle reached its current stage."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "vehicle_id": {

                    "type": "integer",

                    "description":
                        "The vehicle ID to retrieve.",
                }
            },

            "required": [
                "vehicle_id"
            ],

            "additionalProperties": False,
        },
    }
]


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are FleetCare AI, an assistant for fleet maintenance analysis.

You have access to a vehicle analysis tool.

Use get_vehicle_analysis whenever the user asks about a specific
vehicle's prediction, state, risk, trend, deterioration, SHAP
explanation, maintenance priority, or recommended action.

The FleetCare data has three explanation layers:

1. ML prediction
SHAP explains which features contributed to the predicted state.

2. Temporal reasoning
Risk scores, percentiles, persistence and deterioration explain
how the vehicle's risk has behaved over time.

3. Maintenance decision
The rule ID, evidence and reason explain why the system selected
the maintenance action.

Important rules:

- Base vehicle-specific answers on tool results.
- Do not invent vehicle values.
- Do not invent mechanical failures.
- SHAP shows model contribution, not proof of physical failure.
- Clearly distinguish ML prediction from rule-engine action.
- If the requested vehicle does not exist, say so.
- Keep answers concise and understandable to fleet operators, you should not use technical jargon. please use simple language to explain the vehicle's state, risk, and recommended actions.
"""


# ============================================================
# AGENT
# ============================================================

def ask_fleetcare(question: str):

    # ========================================================
    # 1. SEND USER QUESTION TO MODEL
    #
    # Model decides whether a tool is needed.
    # ========================================================

    response = client.responses.create(

        model="gpt-5.6",

        instructions=SYSTEM_PROMPT,

        input=question,

        tools=TOOLS,
    )


    # ========================================================
    # 2. CHECK TOOL CALLS
    # ========================================================

    tool_outputs = []

    for item in response.output:

        if item.type != "function_call":
            continue

        # ----------------------------------------------------
        # get_vehicle_analysis
        # ----------------------------------------------------

        if item.name == "get_vehicle_analysis":

            arguments = json.loads(
                item.arguments
            )

            vehicle_id = arguments[
                "vehicle_id"
            ]

            print(
                f"[Agent] Calling "
                f"get_vehicle_analysis("
                f"vehicle_id={vehicle_id})"
            )

            result = get_vehicle_analysis(
                vehicle_id=vehicle_id,
                engine=engine,
            )

            tool_outputs.append({

                "type":
                    "function_call_output",

                "call_id":
                    item.call_id,

                "output":
                    json.dumps(
                        result,
                        default=str,
                    ),
            })


    # ========================================================
    # 3. NO TOOL REQUIRED
    # ========================================================

    if not tool_outputs:

        return response.output_text


    # ========================================================
    # 4. GIVE TOOL RESULT BACK TO MODEL
    #
    # IMPORTANT:
    #
    # This is the second LLM call.
    #
    # Question
    #   ↓
    # Tool call
    #   ↓
    # DB result
    #   ↓
    # LLM
    #   ↓
    # Final answer
    # ========================================================

    final_response = client.responses.create(

        model="gpt-5.6",

        instructions=SYSTEM_PROMPT,

        previous_response_id=response.id,

        input=tool_outputs,

        tools=TOOLS,
    )


    return final_response.output_text


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk FleetCare: "
    )

    answer = ask_fleetcare(
        question
    )

    print(
        "\nFleetCare AI:\n"
    )

    print(
        answer
    )