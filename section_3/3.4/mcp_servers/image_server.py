import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from mcp.server.fastmcp import FastMCP

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN in .env file")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
)

mcp = FastMCP("ImageGen")


@mcp.tool()
def generate_image(prompt: str, model: str = "black-forest-labs/FLUX.1-schnell") -> str:
    """
    Generate an image from a text prompt using HuggingFace Inference API.
    Return a file path to the saved image.
    """
    print(f"[ImageGen] Generating image for the prompt: {prompt}")

    base_dir = Path(__file__).resolve().parents[3]
    out_dir = Path(os.environ.get("OUT_DIR", str(base_dir / "outputs"))).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "image.png"

    image = client.text_to_image(
        prompt,
        model=model,
    )
    image.save(str(image_path))

    return str(image_path)


if __name__ == "__main__":
    mcp.run(transport="stdio")