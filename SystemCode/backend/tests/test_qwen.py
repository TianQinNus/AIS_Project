from openai import OpenAI

from app.core.config import settings


client = OpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.qwen_base_url,
)


response = client.chat.completions.create(
    model=settings.llm_model,
    messages=[
        {
            "role": "user",
            "content": "Hello, introduce yourself briefly.",
        }
    ],
    stream=False,
    extra_body={
        "enable_thinking": False,
    },
)

print(response.choices[0].message.content)