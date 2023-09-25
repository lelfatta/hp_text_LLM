
# streamlit_layout_app.py

import streamlit as st
import random
import time
import openai
import pandas as pd
import pandasql as psql

#set wide as default
st.set_page_config(layout="wide")

# Access API key from Streamlit secrets and set key 
api_key = st.secrets["openai_api_key"]
openai.api_key = api_key

# Cache data transformation to improve performance
@st.cache_data
def company_rev_rename(companies_df):
    # Rename the 'Revenue (USD millions)' column to 'Revenue'
    companies_df.rename(columns={'Revenue (USD millions)': 'Revenue'}, inplace=True)
    return
#company_rev_rename(companies_df)

# Cache data to improve performance using Streamlit's caching mechanism
@st.cache_data
def load_data(path):
    # Read the CSV file at the given 'path' into a Pandas DataFrame
    df = pd.read_csv(path)
     # Check if the path contains "webscrape.csv"
    if "webscrape.csv" in path:
        company_rev_rename(df)  # Rename column if the condition is met
        
    return df

# Load movies, companies, and music data using the load_data function
# Cached data will be used after the first load
movies_df = load_data("llm_data/movies.csv")  
companies_df = load_data('llm_data/webscrape.csv')  
music_df = load_data('llm_data/musicdata.csv')  


@st.cache_data
def generate_dataframe_metadata(dataframe_dict):
    """
    Generate metadata string for dataframes in the dictionary.
    
    Parameters:
        dataframe_dict (dict): Dictionary of dataframe names and dataframes.
        
    Returns:
        str: Metadata string.
    """
    df_metadata = "Metadata:\n" 
    
    # Loop through each dataframe in the dictionary
    for df_name, df in dataframe_dict.items():
        # Add dataframe name
        df_metadata += f"Dataframe: {df_name}\n"
        
        # Add column names and types
        df_metadata += "Column Names and Types:\n"
        for col, dtype in zip(df.columns, df.dtypes):
            df_metadata += f"  - {col}: {dtype}\n"
        
        # Add sample row format
        #sample_row = df.iloc[0]
        #df_metadata += "Sample Row Format (assume the data values are fake):\n"
        #for col, value in sample_row.items():
         #   df_metadata += f"  - {col}: {type(value).__name__} ({value})\n"
        
        # Add a newline for readability
        df_metadata += "\n"
    
    return df_metadata  # Return the metadata string

# Create a dictionary of dataframes
df_dict = {'companies_df': companies_df, 'movies_df': movies_df, 'music_df': music_df}

# Generate and store metadata
metadata = generate_dataframe_metadata(df_dict)

#takes user input and queries chosen llm to generate sql query
def generate_sql_query(context, prompt):
    response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages= [{"role": "system", "content":"generate ONLY Sql to query relevant data for the User Query. If multiple queries are needed to get all the data, include a --multiple at the very end"},
          {"role": "user", "content": f"{prompt}, \n {context}"}          
       ],
       temperature= .1 
       
    )
    
    return response


context_for_sql = f"{metadata}\nUse like and wildcards on where clauses. The sql's tables will be dataframe names. Use metadata for data context and formatting, show all columns"

#Extracts the table name from the returned SQL query string. Case-insensitive.
def extract_table_from_sql(sql_query):
    # Convert to uppercase to make it case-insensitive
    sql_query_up = sql_query.upper()

    # Find the starting index of 'FROM ' substring
    from_index = sql_query_up.index('FROM ') + 5

    # Extract the string after 'FROM '
    tail_str = sql_query_up[from_index:]

    # Find the first space, or the linebreak to extract the table name
    space_index = tail_str.find(' ')
    linebreak_index = tail_str.find('\n') 
    # Initialize final_index
    final_index = -1

    # Check if either index is -1 (substring not found)
    if space_index == -1 and linebreak_index == -1:
        final_index = len(tail_str)
    elif space_index == -1:
        final_index = linebreak_index
    elif linebreak_index == -1:
        final_index = space_index
    else:
        final_index = min(space_index, linebreak_index)

    # Extract and return table name
    return tail_str[:final_index].strip()

#Executes a SQL query on a Pandas DataFrame specified in a dictionary, returning the result as another DataFrame. Also case-insensitive.
def execute_sql_query(sql_query, df_dict):
 #   if sql_query.endswith("--multiple"):
  #       return "Need to split question into multiple queries to get your data. Please break your question into multiple questions"
    # Extract the table name from the SQL query
    table_in_query = extract_table_from_sql(sql_query).lower()  # Convert to lowercase

    # Convert df_dict keys to lowercase for case-insensitive comparison
    lower_df_dict = {k.lower(): v for k, v in df_dict.items()}

    # Check if the table name exists in the df_dict
    if table_in_query in lower_df_dict:
        # Get the actual dataframe from the dictionary
        target_df = lower_df_dict[table_in_query]

        # Execute the query on the target dataframe
        result_df = psql.sqldf(sql_query)
        return result_df
    else:
        return f"Table {table_in_query} not found."

#convert the dataframe output into markdown for LLM ingestion
#def df_to_markdown(df):
#    return df.to_markdown()

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
sample_movie = movies_df.sample(1)
sample_company = companies_df.sample(1)
sample_music = music_df.sample(1)

# Initialize Streamlit app
def main():
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    st.title('Multi-Dataset Tabular Data Chat App')
    
    #show_sidebar = st.sidebar.checkbox("Show Sidebar", value=True)  # Default set to visible
    #if show_sidebar:
    st.sidebar.title("About This App")
    st.sidebar.markdown("""
    If the chatbot is not working, please reach out to me on my [LinkedIn](www.linkedin.com/in/levielfattal) 
    so I can refill my OpenAI account.
    """)
    #st.sidebar.write("---")  # Horizontal line for clean transition
    st.sidebar.markdown("""
    This app allows you to ask questions about three different datasets spanning three different topics: Movies, Corporations, and Music.
    The app currently uses OpenAI's API to take in your questions, generate a SQL query to get relevant data for, and then use that data to answer your question.
    As of 9/12/23, this app is a barebones proof of concept project. The chatbot does not have any short term memory and will not work well on certain questions. 
    For example, any question that would need multiple queries to answer.
    """)

    st.sidebar.markdown("Here are the Kaggle datasets I am using for this project:")
    st.sidebar.markdown("""
        - [**Movies**](https://www.kaggle.com/datasets/danielgrijalvas/movies)
        - [**Corporations**](https://www.kaggle.com/datasets/claymaker/us-largest-companies)
        - [**Music Sales**](https://www.kaggle.com/datasets/andrewmvd/music-sales)
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
