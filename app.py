import gradio as gr
import os
from groq import Groq
from duckduckgo_search import DDGS

# ======================
# CONFIG
# ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

chat_history = []

# ======================
# FUNCTIONS
# ======================

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No information found."
        summary = ""
        for i, r in enumerate(results, 1):
            summary += f"{i}. {r['title']}\n{r['body']}\n\n"
        return summary.strip()
    except Exception as e:
        return f"Search error: {str(e)}"


def ask_llm(user_text):
    global chat_history

    if not user_text or not user_text.strip():
        return "Please type something.", format_history()

    search_keywords = ["who", "what", "when", "where", "stock", "price", "news", "weather", "current", "latest", "today", "kya", "kaun", "kab", "kahan"]
    need_search = any(word in user_text.lower() for word in search_keywords)

    extra_info = search_web(user_text) if need_search else ""

    messages = [
        {
            "role": "system",
            "content": "You are Lova, a helpful AI created by Saksham Jha. You understand English and Hindi. Keep replies short and natural. Use real-world information when provided."
        }
    ]

    for msg in chat_history[-8:]:
        messages.append(msg)

    if extra_info:
        prompt = f"User: {user_text}\n\nReal-world info:\n{extra_info}\n\nAnswer helpfully:"
    else:
        prompt = user_text

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {str(e)}"

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": reply})

    return reply, format_history()


def format_history():
    if not chat_history:
        return "No messages yet."
    text = ""
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Lova"
        text += f"**{role}:** {msg['content']}\n\n"
    return text


def process_text(user_text):
    reply, history_text = ask_llm(user_text)
    return reply, history_text


def clear_chat():
    global chat_history
    chat_history = []
    return "", "Chat cleared."


# ======================
# INTERFACE
# ======================

with gr.Blocks(title="Lova AI") as demo:
    gr.Markdown("# 🎙️ Lova - Permanent AI")
    gr.Markdown("English + Hindi | Real-world search | Works 24/7")

    history_box = gr.Markdown("No messages yet.")
    text_input = gr.Textbox(label="Type your message", placeholder="Type in English or Hindi...")
    reply_box = gr.Textbox(label="Lova's Reply")

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    send_btn.click(fn=process_text, inputs=text_input, outputs=[reply_box, history_box])
    text_input.submit(fn=process_text, inputs=text_input, outputs=[reply_box, history_box])
    clear_btn.click(fn=clear_chat, outputs=[reply_box, history_box])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)