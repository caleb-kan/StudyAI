import os

from dotenv import load_dotenv

from streamlit_js_eval import get_geolocation

from pymongo.mongo_client import MongoClient

import pandas as pd

import numpy as np

import matplotlib.colors as mcolors

import streamlit as st

import openai

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS

from pydantic import BaseModel, Field

from typing import Type

from bs4 import BeautifulSoup

import requests

import json

import discord
from discord.ext import commands

import aiohttp

import io

from PyPDF2 import PdfReader

from fastapi import FastAPI

import chromadb

import guidance

from filetype import guess

#import nltk

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
discord_api_key = os.getenv("DISCORD_TOKEN")

#defining llm
guidance.llm = guidance.llms.OpenAI(temperature = 0, model = "gpt-3.5-turbo-instruct") 
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
embeddings = OpenAIEmbeddings()
#nltk.download('all')


# 2. Tool for search

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# 3. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("THIS CONTENT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, pass both the objective and url to the function; DO NOT make up any url, the url shall only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    
    def _arun(self, objective: str, url: str):
        raise NotImplementedError("error here")

tools = [
        Tool(
            name="Search",
            func=search,
            description="Useful for when you need to answer questions regarding all topics. You shall ask targeted questions."
        ),
        ScrapeWebsiteTool(),
    ]

memory = ConversationSummaryBufferMemory(
            memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

qa_memory = ConversationSummaryBufferMemory(memory_key="chat_history", return_messages=True, llm=llm, max_token_limit=1000)
chat_history = []

system_message = SystemMessage(
        content=f"""You are a top world class researcher who works for the United States of America and is cleared to receive any internal information on any topic in every company or organization on Earth. You will conduct extremely detailed research on any topic and produce facts based on results; 
        you do not hallucinate, you will try as hard as possible to gather facts & data to back up and prove the research.

        Please make sure you complete the objective above with the following rules:
        1/ You shall do enough research to gather as much information as possible about the objective.
        2/ If there are urls of relevant links & articles, you will scrape it to gather more information about the objective.
        3/ After scraping & search, you shall think "Is there any new information I should search & scrape based on the data I collected to increase the research quality?" If the answer is yes, continue; But do not do this more than 5 iterations.
        4/ You shall not hallucinate, you shall only write facts with the data that you have gathered.
        5/ In the final output, You shall include citations of all reference data & links to back up and prove your research; You shall include citations of all reference data & links to back up and prove your research.
        6/ In the final output, You shall include citations of all reference data & links to back up and prove your research; You shall include citations of all reference data & links to back up and prove your research."""
    )

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs= agent_kwargs,
    memory= memory,
)

def split_string(s, limit):
    lines = s.split("\n")
    strings = []

    for line in lines:
        words = line.split()
        if not words:  # If the line is empty, continue to the next one
            continue
        current_string = words[0]
        for word in words[1:]:
            if len(current_string) + len(word) + 1 > limit:  # +1 for the space
                strings.append(current_string)
                current_string = word
            else:
                current_string += ' ' + word
        strings.append(current_string)  # append the last string

    return strings

def extract_file_content(file_path):
    
    loader = UnstructuredFileLoader(file_path)
        
    documents = loader.load()
    documents_content = '\n'.join(doc.page_content for doc in documents)
    
    return documents_content

text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

def get_doc_search(text_splitter):
    return FAISS.from_texts(text_splitter, embeddings)
    
#short story generator with images
def generate_story(story_idea):
    story = guidance('''
{{#block hidden=True~}}
You are a world class story teller; Your goal is to generate a interesting and captivating short story less than 500 words based on a story idea;

Here is the story idea: {{story_idea}}
Story: {{gen 'story' temperature=0}}
{{/block~}}
     
{{~story~}}
''')
    
    image_url = guidance('''
{{#block hidden=True~}}
You are a world class AI artist, you are tasked to convert text prompts and generate images for the short story; 
Your goal is to generate an amazing text to image prompt and put it in a url that can generate an image from the prompt;
YOU SHALL NOT USE THE SAME IMAGE PROMPT MORE THAN ONCE;

Story: You find yourself standing on the deck of a pirate ship in the middle of the ocean. You are checking if there are still people on the ship
Image url markdown: https://image.pollinations.ai/prompt/a%203d%20render%20of%20a%20man%20standing%20on%20the%20deck%20of%20a%20pirate%20ship%20in%20the%20middle%20of%20the%20ocean
                    
Story: {{story}}
Image url markdown: {{gen 'image_url' temperature=0 max_tokens=500}})
{{~/block~}}

{{~image_url~}}
''')
    
    story = str(story(story_idea=story_idea))
    
    image_url = str(image_url(story = story))
    
    return story, image_url

#discord bot
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

client = commands.Bot(command_prefix = "!", intents = intents)

@client.event
async def on_ready():
    print(f"{client.user} is now running")

@client.command(name='upload')
async def upload(ctx):
    global db
    if not ctx.message.attachments:
        await ctx.send("`Please attach a .pdf file with the !upload command.`")
        return

    file_name = ctx.message.attachments[0].filename
    
    if file_name.endswith(".pdf"):
        await ctx.message.attachments[0].save(fp=format(file_name))
    else:
        await ctx.send("`Please attach a .pdf file with the !upload command.`")
        return
        
    try:
        text_chunks = text_splitter.split_text(extract_file_content(file_name))
        db = get_doc_search(text_chunks)
        await ctx.send("`Text extracted and stored!`")
    except Exception as e:
        print(e)
        await ctx.send("`Failed to extract text!`")
        
    
@client.command(name='research')
async def research(ctx, *, query: str):
    # For now, just echo back the research objective
    try:
        result = agent({"input": query})
        strings = split_string(result["output"], 2000)
        
        for i in range(0, len(strings)):
            await ctx.send(str(strings[i]))
    except Exception as e:
        print(e)
        await ctx.send("`The !research function is limited to queries under 16,384 tokens, please make your research objective shorter and concise.`")
        
@client.command(name='chat')
async def chat(ctx, *, query: str):
    global db

    try:
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever =  db.as_retriever(), memory=qa_memory)
        results = qa({"question": query, "chat_history": chat_history})
        
        strings = split_string(results['answer'], 2000)
        
        for i in range(0, len(strings)):
            await ctx.send(str(strings[i]))
        
    except Exception as e:
        print(e)
        await ctx.send("`Use the !upload function to upload a .pdf or .txt file before chatting!`")
        
@client.command(name='story')
async def story(ctx, *, query: str):
    try: 
        story, image_url = generate_story(query)
        strings = split_string(story, 2000)
        
        for i in range(0, len(strings)):
            await ctx.send(str(strings[i]))
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return await ctx.send('`Could not download image...`')
                data = await resp.read()
                image_stream = io.BytesIO(data)
                await ctx.send(file=discord.File(fp=image_stream, filename="image.png"))
    except Exception as e:
        print(e)
        await ctx.send("`Unable to generate a short story!`")
        
client.run(discord_api_key)

app = FastAPI()
