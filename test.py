import openai
import os

# Fetch the OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Continue with your OpenAI operation
