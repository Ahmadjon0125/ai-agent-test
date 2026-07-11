from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

@tool('execute_python_code')
def execute_python_code(code: str) -> str:
    """Execute Python code and return the output."""
    print("[Tool] execute_python_code called ")
    try:
        # Create a local namespace for executing the code
        local_vars = {}
        exec(code, {}, local_vars)
        return "Code executed successfully. "
    except Exception as e:
        return f"Error executing code: {e}"
    

# llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
llm = ChatOpenAI(
    model="google/gemma-4-31B-it", # Your model name
    openai_api_base="http://192.168.100.54:8000/v1",   # Your server's API endpoint
    openai_api_key="not-needed",
    temperature=0,
    timeout=30,
    max_retries=1
)

agent = create_react_agent(
    model=llm,
    tools=[execute_python_code])


def tester_node(state):
    # !!! TIZIM PROMPTINI SHU YERDA XABARLAR ICHIDA YUBORAMIZ !!!
    system_instruction = (
        "You are a code tester. Execute the provided Python code using the tool. "
        "Verify if the code runs correctly and fulfills the task. "
        "If the code works perfectly and the output is correct, your final response MUST contain the word 'PASS'. "
        "If there is a bug, error, or incorrect output, your final response MUST contain the word 'FAIL'."
    )
    
    # Xabarlar ketma-ketligini yig'amiz
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Test this task/code:\n{state['task']}"}
    ]

    response = agent.invoke({"messages": messages})
    final_message = response['messages'][-1].content

    if "PASS" in final_message.upper():
        result = "PASS"
    elif "FAIL" in final_message.upper():
        result = "FAIL"
    else:
        result = "ERROR"
    
    return {"result" : result, "details": final_message}

