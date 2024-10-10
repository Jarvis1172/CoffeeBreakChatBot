import streamlit as st
import pdfplumber
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO

st.set_page_config(page_title="Coffee Break Assistant", page_icon="☕")

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Folder containing the PDFs and their corresponding text files
PDF_FOLDER_PATH = "C:/Users/ase10774/OneDrive - Aitken Spence/Documents/Branding & Marketing/Coffee Break Chat Bot/PDFS"
TXT_FOLDER_PATH = "C:/Users/ase10774/OneDrive - Aitken Spence/Documents/Branding & Marketing/Coffee Break Chat Bot/TXTS"

# Function to list all PDFs in the folder
def list_pdfs_in_folder(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    pdf_paths = {os.path.splitext(f)[0]: os.path.join(folder_path, f) for f in pdf_files}
    return pdf_paths

# Function to read text from the corresponding .txt file
def read_text_from_txt(pdf_title, txt_folder):
    txt_file_path = os.path.join(txt_folder, pdf_title + ".txt")
    if os.path.exists(txt_file_path):
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return "No text file found for this PDF."

# Function to extract text and links from the PDF using pdfplumber
def get_pdf_text_and_links(pdf_path):
    text_with_links = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_with_links += text

                # Extract links from annotations
                for annotation in page.annots:
                    if annotation.get('uri'):
                        link_text = f"[{annotation['uri']}]"
                        text_with_links += f"\n(Link: {link_text})"
        return text_with_links
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to split text into chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Function to create a vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    {conversation_history}
    
    Answer the question as detailed as possible from the provided context. Provide all details, and if the answer is not in
    the context, say, "answer is not available in the context." Avoid providing incorrect answers.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to save feedback to an Excel file
def save_feedback_to_excel(feedback):
    feedback_df = pd.DataFrame([feedback])  # Convert feedback to DataFrame for easier appending
    file_path = "feedback.xlsx"

    try:
        # Check if the Excel file already exists
        if os.path.exists(file_path):
            existing_df = pd.read_excel(file_path)
            updated_df = pd.concat([existing_df, feedback_df], ignore_index=True)
        else:
            updated_df = feedback_df

        # Save the updated DataFrame back to Excel
        updated_df.to_excel(file_path, index=False)
        st.success("Feedback submitted successfully!")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Instructions page
def instruction_page():
    st.title("Instructions 📖")
    st.markdown("""
    ### Welcome to the Coffee Break Assistant!
    
    This app allows you to interact with documents via a conversational interface. 
    Here's how you can use the app:
    
    1. **Navigate to the Chat Bot page** to interact with the documents.
    2. **Ask questions about the content** of the selected PDF document.
    3. **Provide feedback** on your experience to help us improve!

    ### Tips:
    - Choose a PDF from the list provided.
    - Type your questions in the input box.
    - Provide feedback after your session.

    Enjoy your time with our Coffee Break Assistant! ☕
    """)

# Chat Bot page
def chatbot_page():
    st.title("Chat Bot 🤖")
    st.markdown("### Interact with your Coffee Breaks through our chat interface!")

    # Initialize session state variables
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = ""  # History to maintain conversation context

    if "name" not in st.session_state:
        st.title("👋Hello")
        name = st.text_input("Please enter your name:", key="name_input")
        if st.button("Submit"):
            if name:
                st.session_state.name = name
                st.success(f"Hello, {name}!")
                st.experimental_rerun()  # Reload the app to display the rest of the content
            else:
                st.error("Please enter your name")
    else:
        st.header(f"Welcome to your Coffee Break Assistant, {st.session_state.name}! ☕")

        st.markdown(
            """
            ### Let's Chat Over Coffee! 
            - I'm a brand-new bot, so go easy on me. 😊 
            - Feel free to ask questions about the document, and I'll do my best to serve you some fresh answers!
            """
        )

        # Get PDF files and paths
        PDF_OPTIONS = list_pdfs_in_folder(PDF_FOLDER_PATH)

        # Select the PDF to work with
        selected_pdf_title = st.selectbox("Choose a document to interact with:", list(PDF_OPTIONS.keys()))
        selected_pdf_path = PDF_OPTIONS[selected_pdf_title]

        # Display the content from the corresponding text file
        st.markdown(f"##### Topics from {selected_pdf_title}.pdf")
        pdf_text = read_text_from_txt(selected_pdf_title, TXT_FOLDER_PATH)
        st.text_area(f"{selected_pdf_title}.txt:", pdf_text, height=185)

        # Automatically process the selected PDF without a reload button
        if not os.path.exists("faiss_index") or st.session_state.get("last_loaded_pdf") != selected_pdf_title:
            with st.spinner("Brewing the document, just a moment..."):
                raw_text_with_links = get_pdf_text_and_links(selected_pdf_path)
                if raw_text_with_links:
                    text_chunks = get_text_chunks(raw_text_with_links)
                    if text_chunks:
                        with st.spinner("Pouring over the details, hang tight..."):
                            get_vector_store(text_chunks)
                        st.session_state.last_loaded_pdf = selected_pdf_title
                        st.success(f"{selected_pdf_title} brewed successfully! Ready for your questions. 🍪")
                else:
                    st.error("Failed to process the selected PDF.")
        else:
            st.info("I'm your friendly Coffee Bot, ready to brew some answers for you! ☕")

        # Display previous conversation context
        if st.session_state.conversation_history:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.conversation_history)

        # Chat input for the user
        user_question = st.chat_input("Type your question here...")

        if user_question:
            # Display the user's message in the chat interface
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("Brewing your answer..."):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(user_question)
                
                # Prepare conversation history
                conversation_history = st.session_state.conversation_history

                # Update conversational chain with history
                chain = get_conversational_chain()
                
                response = chain(
                    {"conversation_history": conversation_history, "input_documents": docs, "question": user_question},
                    return_only_outputs=True
                )

                # Get the bot response
                bot_response = response["output_text"]

                # Display the bot response in the chat interface
                with st.chat_message("assistant"):
                    st.markdown(bot_response)

                # Update the conversation history
                st.session_state.conversation_history += f"\nUser: {user_question}\nBot: {bot_response}\n"

        # Feedback Section
        st.markdown(
            """
            ### Enjoyed the chat? We'd love your feedback! 📝
            """
        )

        # Feedback form
        feedback_rating = st.slider("Rate your experience:", min_value=1, max_value=5, step=1)
        problem_text = st.text_area("What went wrong? 🤔", "")
        solution_text = st.text_area("How can we improve? 💡", "")
        feedback_id = f"{selected_pdf_title}_{len(st.session_state.conversation_history)}"

        if st.button("Submit Feedback"):
            feedback = {
                "Feedback ID": feedback_id,
                "Rating": feedback_rating,
                "Problem": problem_text,
                "Solution": solution_text,
            }
            save_feedback_to_excel(feedback)

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = {
    "Instructions": instruction_page,
    "Chat Bot": chatbot_page,
}

selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Run the selected page function
if selected_page:
    pages[selected_page]()
