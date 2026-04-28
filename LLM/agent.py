from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "context.txt")

def read_file(file_path):
	with open(file_path) as file:
		return file.read()


def ask_llm(message, csv_path):
	client = Groq(api_key=os.environ["API_KEY"])

	csv_content = read_file(csv_path)

	chat_completion = client.chat.completions.create(
		messages=[
			{
				"role": "system",
				"content": f"{read_file(CONTEXT_PATH)}\n\nDonnées CSV :\n{csv_content}"
			},
			{
				"role": "user",
				"content": message,
			}
		],
		model="llama-3.3-70b-versatile"
	)

	return chat_completion.choices[0].message.content