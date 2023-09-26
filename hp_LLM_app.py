
# streamlit_layout_app.py

import streamlit as st
import random
import time
import openai
from pypdf import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore



#set wide as default
st.set_page_config(layout="wide")

# Access API key from Streamlit secrets and set key 
api_key = st.secrets["openai_api_key"]
openai.api_key = api_key

#Extract text from pdf using pdfreader
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(open(pdf_path, "rb"))
    text = ""
    #page_num= len(pdf_reader.pages)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text



################################## Still need to add actual response function
#generate the final answer to the user's query 
def generate_final_answer(context, prompt):
    response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages= [{"role": "system", "content":"Use this data to answer the query concisely, but be pleasant. Integrate the query in the answer" },
          {"role": "user", "content": f"{prompt}, \n {context}"}          
       ]
       
    )
    #print(response)
    return response
#########################################################################################################################

# Initialize Streamlit app
def main():
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    st.title('Q&A With Your Printer's User's Manual')
    
    #show_sidebar = st.sidebar.checkbox("Show Sidebar", value=True)  # Default set to visible
    #if show_sidebar:
    st.sidebar.title("About This App")
    st.sidebar.markdown("""
    If the chatbot is not working, please reach out to me on my [LinkedIn](www.linkedin.com/in/levielfattal) 
    so I can refill my OpenAI account.
    """)
    #st.sidebar.write("---")  # Horizontal line for clean transition
    st.sidebar.markdown("""
    This app allows you to fulfill every pre-Y2K office professional's dream: figure out why the printer isn't working! 
    The proof of concept is made specifically to get experience with Langchain's Parent Document Retriever (PDR). The PDR helps maximize the pros and minimize the cons 
    of some of the properties that occur with different size chunking and embedding text data. The idea is that large chunks lose accuracy but maintain context, while small chunks 
    are essentially memorized and lose context, which can hinder creativity in interpretation and a bots's output. I like to make the analogy that small chunks are rote memorization of 
    concepts for a test, while large chunks lose the detail that rote memorization provides they allow a high level overview that is used in life outside of tests. 
    As of 9/27/23, this app is a proof of concept project. The chatbot does not have any short term memory and will not work well on certain questions. The text from the pdf has been minimally 
    processed for the sake of time and the LLM has been instructed to summarize the text as the text data is a bit messy. If the output you receive is not relevant or very strange, please reach out to me 
    on LinkedIn and I will look into it. 
    """)

  st.sidebar.write("---")  # Horizontal line for clean transition

    # Top 1/3 for images
    st.write("###")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Romantisized golden era movie theater screen from a side angle lights dimmed.jfif", caption="Movies by Dalle2", use_column_width=True)
        
    with col2:
        st.image("Corporate buildings in Manhattan from below looking up_.jfif", caption="Corporations by Dalle2", use_column_width=True)
        
    with col3:
        st.image("wide eyed lens shot vinyl record player in focus in the foreground background is a coffee shop but blurred with intense bokeh (1).jfif", caption="Music Sales by Dalle2", use_column_width=True)
                
    with st.expander("See movie sample data"):
        st.write("Use this to help write questions!")
        st.table(sample_movie)

    with st.expander("See company sample data"):
        st.write("Use this to help write questions!")
        st.table(sample_company)

    with st.expander("See music sample data"):
        st.write("Use this to help write questions!")
        st.table(sample_music)
             
         
    # Show 'Clear Chat History' button only if chat history is not empty
    if st.session_state.chat_history:
        if st.button('Clear Chat History'):
            # Clearing session state for chat_history
            st.session_state.chat_history = []
            # Rerun the app to reflect the changes immediately
            st.experimental_rerun()

    # Clean transition
    st.write("---")  # Horizontal line for clean transition
    
   
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input('Type your question here:'):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        #Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

                # Initialize final_answer to store either the generated answer or an error message
        final_answer = ""
        
        # First try-except block for executing the SQL query
        try:
            # Generate SQL query
            sql_query = generate_sql_query(context_for_sql, user_input)
            print(sql_query)
            table_in_query = extract_table_from_sql(sql_query.choices[0].message.content)
            
            # Execute the generated SQL query
            sql_result = execute_sql_query(sql_query.choices[0].message.content, df_dict)
            print(sql_result)
            
        except Exception as e:
            final_answer = f"An error occurred while executing the SQL query. Try rewriting your question to be more specific: {e}"
        
        # Second try-except block for generating the final answer
        if not final_answer:  # Only proceed if no error occurred in the first try-except block
            try:
                # Create final context for prompt (prompt engineering) and generate final answer
                final_context = f"Data: {sql_result}\nBased on this specific data and context, answer the user query."
                final_chat_object = generate_final_answer(sql_result, user_input)
                final_answer = final_chat_object.choices[0].message.content
            
            except Exception as e:
                final_answer = f"An error occurred while generating the final answer: {e}"
     
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response= ""
            response = final_answer  # Placeholder response
            for chunk in response.split():
                
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)   
            
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    
   

# Run the app
if __name__ == '__main__':
    main()
