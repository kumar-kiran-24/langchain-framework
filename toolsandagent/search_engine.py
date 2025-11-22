import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents.factory import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.title("LangChain Search Assistant (Groq + Tools)")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
api=os.getenv("GROK_API_KEY")


prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant with access to three tools ONLY: "
     "search (DuckDuckGo), arxiv, wikipedia. "
     "Never call brave_search or any other tool not explicitly provided.")
])

wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=550)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)

arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=550)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api)

search = DuckDuckGoSearchRun(name='search')

tools = [search, arxiv, wiki]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can search the web using DuckDuckGo, Arxiv & Wikipedia."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask me something...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # LLM
    llm = ChatGroq(
        groq_api_key=api,
        model="llama-3.1-8b-instant",
        streaming=True
    )


    agent = create_agent(
        model=llm,
        tools=tools
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

 
        result = agent.invoke(
            {
                "messages": [
                    {"role": "system", 
                     "content": "Use only search, arxiv, wikipedia. Never use brave_search."},
                    {"role": "user", "content": prompt}
                ]
            },
            callbacks=[st_cb]
        )

        answer = result
        a=result["messages"][-1].content
        st.write(a)

        st.session_state.messages.append(
            {"role": "assistant", "content": a}
        )
