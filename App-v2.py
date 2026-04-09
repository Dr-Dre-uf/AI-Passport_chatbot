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
    page_icon="assistant"
)

# Initialize OpenAI Client
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
else:
    st.error("Please set the OPENAI_API_KEY in your Streamlit secrets.")

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant with expertise in medicine."}
    ]

# Helper: Encode Image
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Core Logic: Generate Response
def generate_response(prompt, image=None, pdf=None):
    # Construct the message content based on input type
    if image:
        base64_image = encode_image(image)
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    elif pdf:
        reader = PdfReader(pdf)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
        user_content = f"User Question: {prompt}\n\nExtracted PDF Content:\n{pdf_text}"
    else:
        user_content = prompt

    # Add the user's message to the session history
    st.session_state.messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            max_tokens=4096,
            temperature=0.3
        )
        assistant_message = response.choices[0].message.content.strip()
        
        # Add assistant response to session history
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    except openai.RateLimitError:
        st.error("Rate limit reached. Please check your OpenAI plan, billing credits, or usage limits.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# UI Layout
st.subheader("MedChat LLM")

with st.sidebar:
    st.subheader("Attach your files")
    uploaded_file = st.file_uploader("Upload an Image or PDF", type=["pdf", "jpg", "jpeg", "png"])
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant with expertise in medicine."}
        ]
        st.rerun()

# User input form
with st.form(key='response_form', clear_on_submit=True):
    user_input = st.text_area('Enter your question here:', placeholder="How can I help you today?", label_visibility='collapsed')
    submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        with st.spinner("Generating response..."):
            if uploaded_file is not None:
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    generate_response(user_input, pdf=uploaded_file)
                elif file_type in ["image/jpeg", "image/png"]:
                    generate_response(user_input, image=uploaded_file)
            else:
                generate_response(user_input)
        st.rerun()

# Display Chat History (Filtering out system messages)
st.markdown("---")
st.markdown("### Chat History")

# Display in reverse order (newest at top) as per your original logic
display_messages = [m for m in st.session_state.messages if m["role"] != "system"]
for message in reversed(display_messages):
    col1, col2 = st.columns([0.8, 0.2])
    
    if message["role"] == "user":
        with col2:
            st.markdown(f"<div style='text-align: right; color: blue;'><b>You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
    elif message["role"] == "assistant":
        with col1:
            st.markdown(f"<div style='color: green;'><b>Assistant 🤖:</b><br>{message['content']}</div>", unsafe_allow_html=True)

# PDF Export Functionality
def create_pdf(chat_text):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Attempt to use Arial if available, otherwise fallback to Helvetica
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
        font_name = "Arial"
    except:
        font_name = "Helvetica"

    y = height - 50
    margin = 50
    line_height = 14
    max_width = width - (2 * margin)

    pdf.setFont(font_name, 10)

    for line in chat_text.splitlines():
        if y < 100:
            pdf.showPage()
            y = height - 50
            pdf.setFont(font_name, 10)
        
        if line.startswith("User:"):
            pdf.setFillColor(colors.blue)
        elif line.startswith("Assistant:"):
            pdf.setFillColor(colors.green)
        else:
            pdf.setFillColor(colors.black)

        # Basic text wrapping logic
        words = line.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if pdf.stringWidth(test_line, font_name, 10) < max_width:
                current_line = test_line
            else:
                pdf.drawString(margin, y, current_line)
                y -= line_height
                current_line = word
        
        pdf.drawString(margin, y, current_line)
        y -= line_height

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

# Export Button
if len(display_messages) > 0:
    chat_string = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in display_messages])
    pdf_data = create_pdf(chat_string)
    st.sidebar.download_button(
        label="Export Conversation as PDF",
        data=pdf_data,
        file_name='chat_history.pdf',
        mime='application/pdf'
    )
