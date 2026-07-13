

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from schema import ResumeExtract

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert HR analyst. Extract structured data from this resume text."),
    ("human", "Resume text:\n```{resume_text}```\n Return a structured  summary with name, summary, years_experience"
    "( float is possible), skills (list of lowercase strings), education (short), recent_companies (list), projects(list) ")
])


# _llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

extract_chain = extract_prompt | _llm.with_structured_output(ResumeExtract)

def extractor_node (state: Dict) -> Dict:
    """
    Input:
        state["resume_text"] -> text
    Output:
        {"extracted": ResumeExtract}
    """
    extracted: ResumeExtract = extract_chain.invoke({"resume_text": state['resume_text']})
    return {"extracted": extracted}

