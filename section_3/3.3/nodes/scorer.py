from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from schema import ResumeExtract, HRDecision

scorer_system = (
    "You are a HR screening assistant. Compare the candidate's extracted resume against the job requirements." \
    "Be strict but fair. Score each area (skilss/experience/education) on 0-100 Decide PASS only overall >= threshold" \
    "and must-have skills are sufficiently met"
)

scorer_prompt = ChatPromptTemplate.from_messages([
    ("system", scorer_system),
    ("human", "Job description: \n```\n{job_description}\n```\n\n Minimum years of experience: {min_years}" \
    "must-have skills: {must_have_skills}\n Nice-to=have skills: {nice_to_have_skills}\n Pass threshold (overall score):" \
    "{threshold} \n\n Extract resume (JSIN) \n{extracted_json} \n\n Return PASS or REJECT with reasons , improvements" \
    "and detailed score breakdown. ")
])

_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

scorer_chain = scorer_prompt | _llm.with_structured_output(HRDecision)

def scorer_node(state:Dict) -> Dict:
    extracted : ResumeExtract = state["extracted"]
    threshold: int= state.get("threshold", 70)
    result: HRDecision = scorer_chain.invoke({
        "job_description" : state["job_description"],
        "min_years" : state["min_years"],
        "must_have_skills": state["must_have_skills"],
        "nice_to_have_skills": state["nice_to_have_skills"],
        "threshold" : threshold,
        "extracted_json": extracted.model_dump_json(indent=2)

    })

    return {
        "decision": result.decision,
        "reasons" : result.reasons,
        "improvements" : result.improvements,
        "score": result.score.model_dump(),
    }

