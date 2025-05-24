import gradio as gr
import os
import datetime
import pandas as pd
import fitz  # PyMuPDF for PDF reading
import openai

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo-16k"

# Log user queries to a file
def log_user_query(user_input):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("user_queries.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] User: {user_input}\n")

# Generate system prompt
def get_system_prompt(mood, length):
    tone = {
        "Friendly": "You are a helpful and friendly assistant.",
        "Professional": "You are a formal and knowledgeable assistant.",
        "Humorous": "You are a witty and engaging assistant."
    }.get(mood, "You are a helpful assistant.")

    detail = {
        1: "Give short and concise answers.",
        2: "Give balanced and informative answers.",
        3: "Give detailed and elaborate explanations."
    }.get(length, "")

    return f"{tone} {detail}"

# Query OpenAI API
def query_openai(message, chat_history, mood, length):
    system_prompt = get_system_prompt(mood, length)
    messages = [{"role": "system", "content": system_prompt}]

    for user, bot in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": message})

    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {e}"

# Handle chat messages
def respond(message, chat_history, mood, length):
    log_user_query(message)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    bot_reply = query_openai(message, chat_history, mood, length)
    chat_history.append((f"{message} ({timestamp})", f"{bot_reply} ({timestamp})"))
    return "", chat_history

# Handle multiple file uploads
def process_file(files, chat_history, mood, length):
    try:
        for file in files:
            if file.name.endswith(".txt"):
                with open(file.name, "r", encoding="utf-8") as f:
                    content = f.read()
                    note = "\n\n[Note: Text was truncated.]" if len(content) > 3000 else ""
                    content = content[:3000]
                prompt = f"Analyze this text file:\n\n{content}{note}"

            elif file.name.endswith(".csv"):
                df = pd.read_csv(file.name)
                summary = df.describe(include='all').to_string()
                sample = df.head(5).to_string(index=False)
                cols = ", ".join(df.columns)
                prompt = (
                    f"This CSV file has {len(df)} rows and {len(df.columns)} columns.\n\n"
                    f"Columns: {cols}\n\nSample rows:\n{sample}\n\nStats:\n{summary}"
                )

            elif file.name.endswith(".pdf"):
                doc = fitz.open(file.name)
                text = ""
                for page in doc:
                    text += page.get_text()
                    if len(text) > 3000:
                        text = text[:3000]
                        break
                prompt = f"Analyze this PDF content:\n\n{text}\n\n[Note: PDF content was truncated.]"

            else:
                chat_history.append((f"[Unsupported file: {file.name}]", "Only .txt, .csv, or .pdf are supported."))
                continue

            response = query_openai(prompt, chat_history, mood, length)
            chat_history.append((f"[Uploaded file: {file.name}]", response))

        return "", chat_history

    except Exception as e:
        return f"Error reading file(s): {e}", chat_history

# Preview uploaded file content
def preview_file(files):
    previews = []
    try:
        for file in files:
            if file.name.endswith(".txt"):
                content = open(file.name, "r", encoding="utf-8").read()
                previews.append(content[:500])
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file.name)
                previews.append(df.head(3).to_string(index=False))
            elif file.name.endswith(".pdf"):
                doc = fitz.open(file.name)
                previews.append(doc[0].get_text()[:500])
            else:
                previews.append("Unsupported file.")
    except Exception as e:
        previews.append(f"Error: {e}")
    return "\n---\n".join(previews)

# Summarize chat history
def summarize_chat(chat_history):
    transcript = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history])
    prompt = f"Summarize this conversation:\n\n{transcript}"
    summary = query_openai(prompt, [], "Professional", 2)
    return summary

# Download chat history
def download_chat(chat_history):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{now}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for user, bot in chat_history:
            f.write(f"User: {user}\nBot: {bot}\n\n")
    return filename

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Smart AI Assistant")

    with gr.Row():
        mood = gr.Dropdown(["Friendly", "Professional", "Humorous"], label="Assistant Mood", value="Friendly")
        length_slider = gr.Slider(1, 3, step=1, value=2, label="Response Detail (1 = Short, 3 = Detailed)")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    file_upload = gr.File(
        label="Upload one or more .txt, .csv, or .pdf files",
        file_types=[".txt", ".csv", ".pdf"],
        file_count="multiple"
    )
    file_preview = gr.Textbox(label="File Preview", lines=8, interactive=False)

    clear = gr.Button("Clear Chat")
    download_btn = gr.Button("Download Chat History")
    summarize_btn = gr.Button("Summarize Chat")

    summary_output = gr.Textbox(label="Chat Summary", lines=5, interactive=False)
    file_output = gr.File()

    state = gr.State([])

    msg.submit(respond, [msg, state, mood, length_slider], [msg, chatbot])
    file_upload.change(preview_file, [file_upload], file_preview)
    file_upload.change(process_file, [file_upload, state, mood, length_slider], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, state])
    download_btn.click(download_chat, [state], file_output)
    summarize_btn.click(summarize_chat, [state], summary_output)

demo.launch()
