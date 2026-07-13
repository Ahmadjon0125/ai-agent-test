from typing import Dict
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader

def loader_node(state: Dict) -> Dict:
    """
    Input:
        state['resume_path'] -> path to PDF or TXT
    Output:
        {'resume_text':"..."}
    """
    path = state['resume_path']
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        docs = PyPDFLoader(path).load()
        text = ' '.join([doc.page_content for doc in docs])
    else:
        docs= TextLoader(path, encoding="utf-8")
        text = ' '.join([doc.page_content for doc in docs])
    
    return {'resume_text': text}