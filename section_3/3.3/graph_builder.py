from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Any, Dict

from nodes.loader import loader_node
from nodes.extractor import extractor_node
from nodes.scorer import scorer_node

class HRState(TypedDict, total=False):
    # Inputs
    resume_path: str
    job_description: str
    min_years: float
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    threshold: int

    # Produced
    resume_text: str
    extracted: Any
    decision: str
    reasons: list[str]
    improvements: list[str]
    score: Dict[str, int]

def build_graph():
    g = StateGraph(HRState)
    g.add_node("loader", loader_node)
    g.add_node("extractor", extractor_node)
    g.add_node("scorer", scorer_node)

    g.add_edge(START, "loader")
    g.add_edge("loader", "extractor")
    g.add_edge("extractor", "scorer")
    g.add_edge("scorer", END)

    return g.compile()