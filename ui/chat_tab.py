"""
Chat tab — multi-turn chat with OSS↔Frontier toggle, RAG grounding, dosage tool, guardrail.
"""
import gradio as gr

from src.agents.memory import ConversationMemory
from src.tools.rag import get_retriever
from src.tools.dosage import convert_dosage, DOSAGE_TOOL_SPEC
from src.guardrails.moderation import moderate, BLOCKED_REPLY

SYSTEM_PROMPT = (
    "You are a helpful medical information assistant. "
    "You answer questions about medications, drug interactions, symptoms, and triage guidance. "
    "Always ground your answers in the provided reference material. "
    "If you are unsure or the information is not in your reference material, say so clearly. "
    "You are NOT a substitute for professional medical advice."
)

_oss_agent = None
_frontier_agent = None
_retriever = None
_memories: dict[str, ConversationMemory] = {}


def _get_agent(model_choice: str):
    global _oss_agent, _frontier_agent
    if model_choice == "OSS (Qwen2.5-0.5B)":
        if _oss_agent is None:
            from src.agents.oss_agent import OSSAgent
            _oss_agent = OSSAgent()
        return _oss_agent, "oss"
    else:
        if _frontier_agent is None:
            from src.agents.frontier_agent import FrontierAgent
            _frontier_agent = FrontierAgent(temperature=0.3)
        return _frontier_agent, "frontier"


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def _maybe_call_dosage_tool(user_message: str) -> str | None:
    """Simple heuristic: if message looks like a dosage conversion, call the tool."""
    msg = user_message.lower()
    if any(kw in msg for kw in ["mg/kg", "convert", "dose in", "dosage of", "how many mg", "mg to g", "g to mg"]):
        # Try to extract value and units — very lightweight
        import re
        # Pattern: "<num> mg/kg"
        m = re.search(r"([\d.]+)\s*mg/kg.*?(\d+)\s*kg", msg)
        if m:
            dose_per_kg = float(m.group(1))
            weight = float(m.group(2))
            res = convert_dosage(dose_per_kg, "mg/kg", weight_kg=weight)
            return f"[Dosage tool result: {res.result}]"
        # Pattern: "<num> mg to g" or "<num> g to mg"
        m2 = re.search(r"([\d.]+)\s*(mg|g|mcg)\s+(?:to|in|into)\s*(mg|g|mcg)", msg)
        if m2:
            val = float(m2.group(1))
            from_u = m2.group(2)
            to_u = m2.group(3)
            res = convert_dosage(val, from_u, to_u)
            return f"[Dosage tool result: {res.result}]"
    return None


def chat_fn(message: str, history: list, model_choice: str, session_id: str) -> tuple[str, list]:
    """Gradio chat function — returns (response, updated_history)."""
    if not message.strip():
        return "", history

    # Input moderation
    input_mod = moderate(message)
    if not input_mod.safe:
        blocked = f"[Input blocked: {input_mod.category}. Please ask a different question.]"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": blocked},
        ]
        return "", history

    agent, agent_key = _get_agent(model_choice)
    mem_key = f"{session_id}_{agent_key}"
    if mem_key not in _memories:
        _memories[mem_key] = ConversationMemory(system_prompt=SYSTEM_PROMPT)
    memory = _memories[mem_key]

    # RAG retrieval
    retriever = _get_retriever()
    rag_context = retriever.format_context(message, top_k=3)

    # Dosage tool
    tool_result = _maybe_call_dosage_tool(message)

    # Assemble augmented user message
    augmented = message
    if rag_context:
        augmented = f"{rag_context}\n\nUser question: {message}"
    if tool_result:
        augmented += f"\n\n{tool_result}"

    memory.add("user", augmented)
    messages = memory.messages()

    try:
        response = agent.chat(messages)
    except Exception as e:
        response = f"[Error: {e}]"
        memory.add("assistant", response)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return "", history

    # Output moderation
    output_mod = moderate(message, response)
    if not output_mod.safe:
        response = BLOCKED_REPLY

    memory.add("assistant", response)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", history


def clear_fn(session_id: str, model_choice: str):
    """Clear conversation memory for the current model."""
    _, agent_key = _get_agent(model_choice)
    mem_key = f"{session_id}_{agent_key}"
    if mem_key in _memories:
        _memories[mem_key].clear()
    return []


def build_chat_tab():
    import uuid
    session_id = gr.State(str(uuid.uuid4()))

    with gr.Column():
        gr.Markdown(
            "## Medical AI Assistant\n"
            "Ask about medications, dosing, drug interactions, allergies, and symptom triage. "
            "Answers are grounded in 5 medical reference documents.\n\n"
            "> **Disclaimer:** This is a research demonstration, not medical advice. "
            "Always consult a qualified healthcare professional."
        )
        model_choice = gr.Radio(
            choices=["OSS (Qwen2.5-0.5B)", "Frontier (Gemini 2.0 Flash)"],
            value="Frontier (Gemini 2.0 Flash)",
            label="Model",
        )
        chatbot = gr.Chatbot(height=500, label="Conversation")
        msg_box = gr.Textbox(
            placeholder="Ask a medical question… (e.g. 'What is the max dose of paracetamol?')",
            label="Your message",
            lines=2,
        )
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear conversation")

    send_btn.click(
        chat_fn,
        inputs=[msg_box, chatbot, model_choice, session_id],
        outputs=[msg_box, chatbot],
    )
    msg_box.submit(
        chat_fn,
        inputs=[msg_box, chatbot, model_choice, session_id],
        outputs=[msg_box, chatbot],
    )
    clear_btn.click(
        clear_fn,
        inputs=[session_id, model_choice],
        outputs=[chatbot],
    )
