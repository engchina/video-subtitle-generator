from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# read local .env file
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]

llm = ChatOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model_name=OPENAI_MODEL_NAME, temperature=0)


def translate_subtitles(input_srt, output_srt, prompt, source_lang='Chinese', target_lang='English'):
    prompt = PromptTemplate(
        input_variables=["source_lang", "target_lang", "text"],
        template=prompt
    )

    chain = prompt | llm | StrOutputParser()

    with open(input_srt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []
    for line in tqdm(lines):
        if line.strip() and not line[0].isdigit() and "-->" not in line:
            translated_text = chain.invoke(
                input={"source_lang": source_lang, "target_lang": target_lang, "text": line.strip()})
            translated_lines.append(translated_text + "\n")
        else:
            translated_lines.append(line)

    with open(output_srt, "w", encoding="utf-8") as f:
        f.writelines(translated_lines)
