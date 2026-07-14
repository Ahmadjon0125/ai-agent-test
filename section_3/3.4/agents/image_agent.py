import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


async def setup_image_agent():
    base_dir = Path(__file__).resolve().parents[3]

    server_path = base_dir / "section_3" / "3.4" / "mcp_servers" / "image_server.py"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OUT_DIR"] = str(output_dir)

    client = MultiServerMCPClient({
        "image": {
            "command": sys.executable,
            "args": [str(server_path)],
            "transport": "stdio",
            "env": {**os.environ},
        }
    })
    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="google/gemma-4-31B-it", # Your model name
        openai_api_base="http://192.168.100.54:8000/v1",   # Your server's API endpoint
        openai_api_key="not-needed",
        temperature=0,
        timeout=30,
        max_retries=1
    )
    # llm = ChatGroq(
    #     model="meta-llama/llama-4-scout-17b-16e-instruct",
    #     temperature=0.4)
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)
    return create_react_agent(llm, tools)


async def image_node(state):
    image_agent = await setup_image_agent()
    response = await image_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": f"Generate a thumbnail image for this LinkedIn post: \n {state['post_text']}"
        }]
    })

    content = response["messages"][-1].content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        content = "\n".join(text_parts)

    if isinstance(content, str):
        path = content.strip()
        if path and os.path.exists(path):
            return {"image_path": path}

    base_dir = Path(__file__).resolve().parents[3]
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "image.png"
    if image_path.exists():
        return {"image_path": str(image_path)}

    return {"image_path": ""}