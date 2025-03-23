# summarizer.py
import requests
import json

class Summarizer:
    def __init__(self, url="http://localhost:11434/api/generate"):
        self.url = url

    def generate_summary(self, context, sources, prompt="Briefly summarize this text:"):
        source_hint = f"From {', '.join(sources)}: " if sources else ""
        full_prompt = f"{source_hint}{prompt} {context}"
        response = requests.post(
            self.url,
            json={"model": "mistral:7b-instruct-q4_0", "prompt": full_prompt, "max_tokens": 50, "temperature": 0.5, "top_p": 0.9},
            stream=True
        )
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    break
        return full_response.strip()