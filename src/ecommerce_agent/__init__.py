from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable


MAX_ITERATIONS = 10
MODEL = "gpt-5.6-luna"


# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"get_product_price {product}")
    prices = {"laptop": 1299.999, "headphones": 149.5}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price."""
    print(f"apply_discount {discount_tier} to {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount /100), 2)
    
# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]

    tools_dict = { t.name: t for t in tools }

    llm = init_chat_model(f"{MODEL}", temperature=0, use_responses_api=True)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpfun shopping assistant "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real prince.\n"
                "2. Only call apply_discount AFTER you have received "
                " a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use. - do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]

    # Thought process
    for iteration in range(1, MAX_ITERATIONS +1):
        print(f"Iteration {iteration}")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            return ai_message.content

        # Process only the first TOOL call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR:")
    return None


def main() -> None:
    print("Hello from ecommerce-agent!")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print("="*60)
    print(result)

    return result


if __name__ == "__main__":
  main()