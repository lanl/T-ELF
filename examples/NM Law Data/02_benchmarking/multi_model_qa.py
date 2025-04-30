import openai
import anthropic
import google.generativeai as genai
from ollama import Client

class MultiModelLegalQA:
    def __init__(self, openai_api_key, anthropic_api_key, google_bard_api_key,
                 ollama_host="http://localhost:11434"):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.google_bard_api_key = google_bard_api_key
        self.ollama_host = ollama_host

    def format_prompt(self, source: str, question: str) -> str:
        return (
            "You are an expert lawyer. Use the following source text to answer the question clearly and precisely.\n\n"
            f"SOURCE:\n{source}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER AS A LAWYER:"
        )

    def query_openai(self, question, model="gpt-4o-2024-11-20"):
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content.strip()

    def query_claude_opus(self, question):
        client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": question}],
        )
        return message.content[0].text.strip()

    def query_google_bard(self, question):
        try:
            genai.configure(api_key=self.google_bard_api_key)
            model = genai.GenerativeModel("gemini-2.5-pro-preview-03-25")
            response = model.generate_content(question)
            return response.text.strip()
        except Exception as e:
            return f"[Gemini API Error] {str(e)}"

    def query_ollama(self, question, ollama_model):
        try:
            client = Client(host=self.ollama_host)
            response = client.chat(model=ollama_model, messages=[{"role": "user", "content": question}])
            return response['message']['content'].strip()
        except Exception as e:
            return f"[Ollama Error - {ollama_model}] {str(e)}"

    def ask_all_models(self, source, question):
        prompt = self.format_prompt(source, question)
        responses = {}

        responses['GPT-4o'] = self.query_openai(prompt, model="gpt-4o-2024-11-20")
        responses['GPT-3.5'] = self.query_openai(prompt, model="gpt-3.5-turbo-1106")
        responses['Claude 3 Opus'] = self.query_claude_opus(prompt)
        responses['Gemini Pro'] = self.query_google_bard(prompt)
        responses['LLaMA 3.1–70B'] = self.query_ollama(prompt, ollama_model="llama3.2:3b-instruct-fp16")
        responses['Nemotron–70B-Instruct'] = self.query_ollama(prompt, ollama_model="nemotron:70b-instruct-q8_0")

        return responses
