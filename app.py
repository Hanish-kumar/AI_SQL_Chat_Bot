from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

def init_database(user, password, host, port, database) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="mistral")

    def get_schema(_):
        return db.get_table_info()
    
    return(
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template="""
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="mistral")

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your Database assistant. Ask me anything about your Database")
    ]



st.set_page_config(page_title="Chat with Database", page_icon=":speech_balloon:")

st.title("Chat with Database")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is chat application using Database. Connect to the Database and assist yourself")
    st.text_input("host", value="http://sql12.freesqldatabase.com")
    st.text_input("port", value="3306")
    st.text_input("user", value="sql12779393")
    st.text_input("password", type="password", value="vydhyala@123")
    st.text_input("database", value="sql12779393")
    if st.button("Connect"):
        with st.spinner("Connecting the Database..."):
            try:
                db = init_database(user, password, host, port, database)
                st.session_state.db = db
                st.success("Connected to Database!")
            except Exception as e:
                st.error(f"Connection failed: {e}")


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a text...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Check if DB is connected before querying
    if "db" in st.session_state:
        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
        st.session_state.chat_history.append(AIMessage(content=response))
    else:
        with st.chat_message("AI"):
            st.markdown("⚠️ Please connect to the database first using the sidebar.")
