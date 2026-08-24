# ecommerce-agent

A minimal shopping-assistant agent built with LangChain, following the **ReAct** (Reason + Act) pattern: it can look up a product's price and apply a discount tier, deciding for itself which tool to call and in what order, with every run traced to LangSmith.

All of the logic lives in [src/ecommerce_agent/__init__.py](src/ecommerce_agent/__init__.py).

## Setup

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY, LANGSMITH_API_KEY, etc.
uv run ecommerce-agent
```

## Concepts

### The ReAct loop

ReAct stands for **Reason, then Act**: instead of answering in one shot, the model alternates between *thinking* about what it needs and *acting* by calling a tool to get real information, then reasons again over the result. `run_agent` implements this as an explicit loop:

```python
for iteration in range(1, MAX_ITERATIONS + 1):
    ai_message = llm_with_tools.invoke(messages)

    if not ai_message.tool_calls:
        return ai_message.content        # model is done reasoning — final answer

    tool_call = ai_message.tool_calls[0]  # model chose to act
    observation = tools_dict[tool_call["name"]].invoke(tool_call["args"])

    messages.append(ai_message)
    messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    # loop again — the model now reasons with the observation in context
```

Each iteration:

1. The LLM sees the full conversation so far (system prompt + question + any prior tool results).
2. It either **responds** (no `tool_calls` → we return the answer) or **acts** (emits a `tool_calls` entry naming a tool and arguments).
3. If it acted, the code runs that tool and appends the result as a `ToolMessage` — this is the "observation" step.
4. The loop repeats, so the next reasoning step has the observation available.

`messages` is the agent's working memory — it's what turns a single stateless LLM call into a multi-step reasoning process. `MAX_ITERATIONS` (10) is a safety valve against infinite loops; if the model never settles on a final answer, `run_agent` gives up and returns `None`.

This implementation deliberately processes only `tool_calls[0]` per iteration, forcing one tool call per turn even if the model requested several — that's why the system prompt spells out an explicit order ("call `get_product_price` before `apply_discount`") rather than relying on the model to batch calls correctly.

### `@tool` — turning a function into something the model can call

```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    ...
```

LangChain's `@tool` decorator wraps a plain Python function into a `Tool` object the LLM can be told about:

- The **type hints** (`product: str`) become a JSON schema describing the expected arguments.
- The **docstring** becomes the tool's description — this is literally what the model reads to decide *when* to use it, so it doubles as documentation and as a prompt.
- The function itself becomes invocable through a consistent `.invoke(args_dict)` interface, regardless of what the underlying function's own signature looks like.

`llm.bind_tools(tools)` hands these schemas to the model. From that point on, the model can either reply with plain text or reply with a structured request to call one of these tools — it never sees or runs the Python code, only the name and the schema.

### `@traceable` — LangSmith tracing

```python
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    ...
```

`@traceable` (from the `langsmith` package) wraps a function so every call — its inputs, outputs, exceptions, and timing — is recorded as a trace, as long as `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set (see `.env.example`). Because `run_agent` itself is decorated, and it calls `llm_with_tools.invoke(...)` and `tool_to_use.invoke(...)` internally, LangSmith automatically nests those calls into one tree per run: you get the full sequence of LLM calls, the exact tool arguments chosen, and each tool's return value, viewable in the LangSmith UI under the run named `"LangChain Agent Loop"`. This is what makes a black-box "why did the agent do that" question answerable — you can open a specific run and see each reasoning/acting step in order.

### Callable functions — two different meanings of "call"

It's easy to conflate two distinct things happening in this code, both called "calling a function":

1. **Python-callable.** `get_product_price` and `apply_discount` are still ordinary Python functions under the `@tool` wrapper — nothing stops you from calling their logic directly in code.
2. **Model-callable (tool/function calling).** Once bound via `bind_tools`, the *model* can request that one of these functions be run, by returning a structured `tool_calls` entry — a name plus JSON arguments — instead of free text. The model never executes anything itself; it only asks.

The bridge between the two is this line:

```python
tools_dict = {t.name: t for t in tools}
...
tool_to_use = tools_dict.get(tool_name)
observation = tool_to_use.invoke(tool_args)
```

Your code is the only thing that ever actually calls the function — it looks up the name the model requested in `tools_dict` and invokes the matching tool with the arguments the model supplied. This indirection is what makes tool use safe and predictable: the model can only ever request calls to functions you explicitly exposed, with arguments validated against the schema `@tool` generated, and your code stays in control of what actually executes.

## Project structure

```
src/ecommerce_agent/__init__.py   # tools, ReAct loop, and entrypoint
.env.example                       # required environment variables
pyproject.toml                     # dependencies (uv-managed)
```
