import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# uses gemini LLM to optimize the user query. 
def optimize_query(query, system_prompt=None, model_name='models/gemini-1.5-flash'):
    """
    Uses Gemini 1.5 Flash (generation model) to optimize/refine the user query for better retrieval.
    Optionally accepts a system prompt for context.
    """
    prompt = f"Optimize the following query for information retrieval: {query}"
    if system_prompt:
        prompt = system_prompt + "\n" + prompt
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip() 
