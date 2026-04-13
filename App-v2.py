import streamlit as st
import openai
from openai import OpenAI
import io
import os
import base64
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="AI Passport Chatbot",
    page_icon=None
)

# --- SECURITY & PRIVACY BANNER ---
st.warning(
    "**Privacy and Data Safety:** Please ensure that any uploaded images or PDFs do not contain "
    "Personally Identifiable Information (PII) such as full names, home addresses, or government IDs. "
    "Always de-identify records before uploading."
)

# Initialize NaviGator Toolkit Client
if 'NAVIGATOR_TOOLKIT_API_KEY' in st.secrets:
    os.environ['NAVIGATOR_TOOLKIT_API_KEY'] = st.secrets['NAVIGATOR_TOOLKIT_API_KEY']
    client = OpenAI(
        api_key=os.environ['NAVIGATOR_TOOLKIT_API_KEY'],
        base_url="https://api.ai.it.ufl.edu/v1"
    )
else:
    st.error("Missing API Key. Please add 'NAVIGATOR_TOOLKIT_API_KEY' to your Streamlit Secrets.")
    st.stop()

# Initialize session state for messages and errors
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant with expertise in medicine. Provide clear, accurate information based on the context provided."}
    ]
    # No pre-filled assistant message in data history to ensure strict role alternation (system -> user -> assistant)

if "error" not in st.session_state:
    st.session_state.error = None

# Helper: Encode Image
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Core Logic: Generate Response
def generate_response(prompt, image=None, pdf=None):
    if image:
        base64_image = encode_image(image)
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    elif pdf:
        reader = PdfReader(pdf)
        pdf_text = "".join([page.extract_text() for page in reader.pages])
        user_content = f"User Question: {prompt}\n\nDocument Context:\n{pdf_text}"
    else:
        user_content = prompt

    st.session_state.messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="gemma-3-27b-it",
            messages=st.session_state.messages,
            max_tokens=4096,
            temperature=0.3
        )
        assistant_reply = response.choices[0].message.content.strip()
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    except openai.RateLimitError:
        st.session_state.error = "Rate limit reached. Please check your NaviGator Toolkit account limits."
        return None
    except Exception as e:
        st.session_state.error = f"Error calling model: {e}"
        return None

# Sidebar Layout
with st.sidebar:
    st.title("Settings")
    st.info("Model: gemma-3 (NaviGator AI)")
    
    st.divider()
    
    st.subheader("Upload Center")
    uploaded_file = st.file_uploader("Upload Image or PDF", type=["pdf", "jpg", "jpeg", "png"])
    
    st.divider()
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant with expertise in medicine."}
        ]
        st.session_state.error = None
        st.rerun()

# Main UI
st.title("NaviGator Chatbot")

# User Input Form
with st.container():
    with st.form(key='response_form', clear_on_submit=True):
        user_input = st.text_area('Input', placeholder="Type your question here...", label_visibility='collapsed')
        submit_button = st.form_submit_button("Submit Request")

        if submit_button and user_input:
            st.session_state.error = None # Clear previous errors
            with st.spinner("Processing..."):
                if uploaded_file:
                    file_type = uploaded_file.type
                    if file_type == "application/pdf":
                        generate_response(user_input, pdf=uploaded_file)
                    else:
                        generate_response(user_input, image=uploaded_file)
                else:
                    generate_response(user_input)
            st.rerun()

# Error Display (Persistently shown across reruns if present)
if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Dismiss Error"):
        st.session_state.error = None
        st.rerun()

# Chat Display
st.markdown("### Conversation History")

visible_history = [m for m in st.session_state.messages if m["role"] != "system"]

# Show a greeting if the history is empty
if not visible_history:
    with st.chat_message("assistant"):
        st.markdown("Hello! I am your NaviGator AI assistant. How can I help you today?")

for message in visible_history:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, list):
            # Handle mixed content (images + text)
            for item in content:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"])
        else:
            st.markdown(content)

# PDF Export Logic
def create_pdf(chat_text):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    p.setFont("Helvetica", 10)
    y = height - 50
    margin = 50
    line_height = 14
    max_w = width - (2 * margin)

    for line in chat_text.splitlines():
        if y < 100:
            p.showPage()
            y = height - 50
            p.setFont("Helvetica", 10)
        
        if line.startswith("User:"): p.setFillColor(colors.blue)
        elif line.startswith("Assistant:"): p.setFillColor(colors.green)
        else: p.setFillColor(colors.black)

        words = line.split()
        curr = ""
        for word in words:
            if p.stringWidth(f"{curr} {word}", "Helvetica", 10) < max_w:
                curr = f"{curr} {word}".strip()
            else:
                p.drawString(margin, y, curr)
                y -= line_height
                curr = word
        p.drawString(margin, y, curr)
        y -= line_height

    p.save()
    buffer.seek(0)
    return buffer.getvalue()

# Sidebar Export Button
if len(visible_history) > 0:
    export_lines = []
    for m in visible_history:
        role = m['role'].capitalize()
        content = m['content'][0]['text'] if isinstance(m['content'], list) else m['content']
        export_lines.append(f"{role}: {content}")
    
    chat_str = "\n".join(export_lines)
    pdf_bin = create_pdf(chat_str)
    
    st.sidebar.download_button(
        label="Download History (PDF)",
        data=pdf_bin,
        file_name='medchat_history.pdf',
        mime='application/pdf',
        use_container_width=True
    )
