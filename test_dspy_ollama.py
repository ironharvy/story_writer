import litellm
import traceback

try:
    response = litellm.completion(
        model="ollama/qwen3",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="http://localhost:11434",
        api_key=""
    )
    print(response)
except Exception as e:
    traceback.print_exc()
