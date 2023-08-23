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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
discord_api_key = os.getenv("DISCORD_TOKEN")

#defining llm
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

#text data storage
vector_db = []

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

def vector_store(file_name, attachment):
    if file_name.endswith('.txt'):
        raw_documents = TextLoader(attachment).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        return db
    else:
        raw_documents = PyPDFLoader(attachment).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        return db

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
        await ctx.send("Please attach a .txt or .pdf file with the `!upload` command.")
        return

    attachment = ctx.message.attachments[0]
    attachment_url = attachment.url
    file_name = attachment.filename
    print(file_name)
        
    if file_name.endswith(('.txt', '.pdf')):
        try:
            db = vector_store(file_name, attachment_url)
            await ctx.send("Text extracted and stored!")
        except Exception as e:
            print(e)
            await ctx.send("Failed to extract text!")
    else:
        await ctx.send("Please upload a .txt or .pdf file!")
        
@client.command(name='research')
async def research(ctx, *, query: str):
    # For now, just echo back the research objective
    try:
        result = agent({"input": query})
        await ctx.send(result["output"])
    except Exception as e:
        print(e)
        await ctx.send("The !research function is limited to queries under 16,384 tokens, please make your research objective shorter and concise.")
        
@client.command(name='chat')
async def chat(ctx, *, query: str):
    try:
        global db
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":1})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True, verbose = True)
        answer = qa({"query": query})
        await ctx.send(answer['result'])
    except Exception as e:
        print(e)
        await ctx.send("Use the !upload function to upload a .pdf or .txt file before chatting!")

client.run(discord_api_key)

app = FastAPI()