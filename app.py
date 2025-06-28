import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


st.set_page_config(page_title="Startup Idea Generator", layout="centered")
st.title("🚀 AI Startup Idea Generator")
st.markdown("Powered by [LangChain + Google Gemini]")


load_dotenv()


llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0.9
)


prompt_domain = PromptTemplate(
    input_variables=["domain"],
    template="I want to open a startup in the {domain} domain. Suggest a few innovative startup ideas."
)

prompt_idea = PromptTemplate(
    input_variables=["idea"],
    template="Here is a startup idea: {idea}. Suggest 4-5 name ideas that are creative and relevant to the idea."
)

prompt_pitch = PromptTemplate(
    input_variables=["idea"],
    template="Here is a startup idea: {idea}. Suggest 4-5 impressive and concise elevator pitches for it."
)

prompt_mvp = PromptTemplate(
    input_variables=["idea"],
    template="Here is a startup idea: {idea}. Suggest 4-5 MVP features to include in its first version."
)


domain_chain = LLMChain(llm=llm, prompt=prompt_domain, output_key="idea")
name_chain = LLMChain(llm=llm, prompt=prompt_idea, output_key="name")
pitch_chain = LLMChain(llm=llm, prompt=prompt_pitch, output_key="pitch")
mvp_chain = LLMChain(llm=llm, prompt=prompt_mvp, output_key="mvp")


full_chain = SequentialChain(
    chains=[domain_chain, name_chain, pitch_chain, mvp_chain],
    input_variables=["domain"],
    output_variables=["idea", "name", "pitch", "mvp"],
    verbose=False
)


domain_input = st.text_input("Enter a startup domain (e.g. finance, edtech, health, defence)")

if st.button("Generate Startup Plan"):
    if not domain_input.strip():
        st.warning("Please enter a valid domain.")
    else:
        with st.spinner("Generating startup ideas..."):
            result = full_chain({"domain": domain_input.strip()})

        
        st.markdown("## 🚀 <span style='font-size:32px'>AI Startup Plan</span>", unsafe_allow_html=True)

        
        st.markdown("### 💡 <span style='font-size:26px'>Startup Idea</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px; font-weight:600'>{result['idea']}</div>", unsafe_allow_html=True)

       
        st.markdown("### 🏷️ <span style='font-size:22px'>Name Suggestions</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px; font-weight:600'>{result['name']}</div>", unsafe_allow_html=True)

        
        st.markdown("### 🎯 <span style='font-size:22px'>Elevator Pitches</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px; font-weight:600'>{result['pitch']}</div>", unsafe_allow_html=True)

      
        st.markdown("### 🛠️ <span style='font-size:22px'>MVP Feature List</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px; font-weight:600'>{result['mvp']}</div>", unsafe_allow_html=True)
