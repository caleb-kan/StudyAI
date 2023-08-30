# Introducing StudyAI, a Discord Study Bot 
Welcome to the official documentation repository for our Discord bot! This space is dedicated to providing comprehensive guides and insights into the bot's functionalities and how to make the most of them. 
## Table of Contents  
- [Overview](#overview) 
- [Getting Started](#getting-started) 
	- [Installation](#installation) 
	- [Technical Initialization](#technical-initialization)  	
	- [User Commands](#user-commands) 
- [Feedback and Contributions](#feedback-and-contributions) 
## Overview 
This Discord bot has been designed with both users and administrators in mind, offering a suite of features to enhance server management, user engagement, and overall experience. Whether you're a server owner, a moderator, or just an everyday user, this documentation will guide you through all the features and commands our bot offers. 
## Getting Started  
### Installation 
Connecting our Discord bot to your server is a straightforward process. Simply click on [this link](https://discord.com/api/oauth2/authorize?client_id=1138347327866818662&permissions=534723950656&scope=bot) to initiate the installation. This will redirect you to Discord's official OAuth2 authorization page. Here, you'll be prompted to choose a server where you'd like to add the bot. Ensure that you have the necessary permissions on the server to invite bots. Once you've selected the desired server, follow the on-screen instructions to complete the bot's installation. After a successful installation, the bot will be active and ready to serve on your server!

### Technical Initialization
- Libaries to install: 
		- `aiohttp`, `beautifulsoup4`, `discord.py`, `fastapi`, `langchain`, `matplotlib`, `numpy`, `openai`, `pandas`, `pydantic`, `pymongo`, `PyPDF2`, `python-dotenv`, `Requests`, `streamlit`, `streamlit_js_eval`, `pypdf`, `redis`, `guidance`, `sentence_transformers`, `unstructured`, `chromadb`, `redis_om` 
- Defining Large Language Models
	```python
	guidance.llm  =  guidance.llms.OpenAI("text-davinci-003")
	llm  =  ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
	embeddings  =  SentenceTransformerEmbeddings(model_name  =  "all-MiniLM-L6-v2")
	```

### User Commands 
The bot offers a set of commands tailored to enhance user experience, with four primary commands at its core. Below is a detailed breakdown of these commands:

1.  **!research**:
    -   **Usage**: `!research [research objective]`
    -   **Description**: When you invoke the `!research` command followed by a research objective, the bot embarks on a mission across the web. It dives into the top search results related to your query. Using a sophisticated large language model, the bot evaluates whether a website contains information relevant to your research objective. If a match is found, the bot extracts the necessary text data. This process is repeated for multiple websites, ensuring a comprehensive search. At the end of this research expedition, the bot crafts a summarized output of its findings, neatly packaged with citations for your reference.
    - **Technical**: 
		- `function to search the web`
			 ```python
		    def  search(query):
				url  =  "https://google.serper.dev/search"
				payload  =  json.dumps({
					"q": query
				})
		
				headers  = {
					'X-API-KEY': serper_api_key,
					'Content-Type': 'application/json'
				}

				 response  =  requests.request("POST", url, headers=headers, data=payload)
				 
				print(response.text)
				
				return  response.text 
			```
			
		- `function to scrape website`
			```python 
			def  scrape_website(objective: str, url: str):
				
				print("Scraping website...")
				
				headers  = {
				'Cache-Control': 'no-cache',
				'Content-Type': 'application/json',
				}

				data  = {
					"url": url
				}

				data_json  =  json.dumps(data)

				post_url  =  f"https://chrome.browserless.io/content?token={browserless_api_key}"

				response  =  requests.post(post_url, headers=headers, data=data_json)

				if  response.status_code  ==  200:

					soup  =  BeautifulSoup(response.content, "html.parser")

					text  =  soup.get_text()

					print("THIS CONTENT:", text)

				  

					if  len(text) >  10000:

						output  =  summary(objective, text)

						return  output

					else:

						return  text

				else:

					print(f"HTTP request failed with status code {response.status_code}")
			
			```
		- `function to produce final summary`
			```python 
			def  summary(objective, content):

				text_splitter  =  RecursiveCharacterTextSplitter(
					separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)

				docs  =  text_splitter.create_documents([content])

				map_prompt  =  """
				Write a summary of the following text for {objective}:
				"{text}"
				SUMMARY:
				"""
				
				map_prompt_template  =  PromptTemplate(
					template=map_prompt, input_variables=["text", "objective"])

				summary_chain  =  load_summarize_chain(
					llm=llm,
					chain_type='map_reduce',
					map_prompt=map_prompt_template,
					combine_prompt=map_prompt_template,
					verbose=True
				)
				
				output  =  summary_chain.run(input_documents=docs, objective=objective)
				
				return  output
			```
		- `creating a research AI agent using the functions above with LangChain & OpenAI`
			```python 
			class  ScrapeWebsiteInput(BaseModel):
				"""Inputs for scrape_website"""
				objective: str  =  Field(
					description="The objective & task that users give to the agent")
				url: str  =  Field(description="The url of the website to be scraped")
			```
			```python 
			class  ScrapeWebsiteTool(BaseTool):
				name  =  "scrape_website"
				description  =  "useful when you need to get data from a website url, pass both the objective and url to the function; DO NOT make up any url, the url shall only be from the search results"
				args_schema: Type[BaseModel] =  ScrapeWebsiteInput

				def  _run(self, objective: str, url: str):
					return  scrape_website(objective, url)
					
				def  _arun(self, objective: str, url: str):
					raise  NotImplementedError("error here")
			```
			```python 
			tools  =  [
					Tool(
					name="Search",
					func=search,
					description="Useful for when you need to answer questions regarding all topics. You shall ask targeted questions."
					),
					ScrapeWebsiteTool(),
				]
			```
			```python
			memory  =  ConversationSummaryBufferMemory(
				memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)
			```
			```python
			system_message  =  SystemMessage(
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
			```
			```python 
			agent_kwargs  = {
				"extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
				"system_message": system_message,
			}
			```
			```python 
			agent  =  initialize_agent(
				tools,
				llm,
				agent=AgentType.OPENAI_FUNCTIONS,
				verbose=True,
				agent_kwargs=  agent_kwargs,
				memory=  memory,
			)
			```
		- `!research command`
			```python
			@client.command(name='research')
			async  def  research(ctx, *, query: str):
				try:
					result  =  agent({"input": query})
					strings  =  split_string(result["output"], 2000)
					
					for  i  in  range(0, len(strings)):
						await  ctx.send(str(strings[i]))
				except  Exception  as  e:
					print(e)
					await  ctx.send("The !research function is limited to queries under 16,384 tokens, please make your research objective shorter and concise.")
			``` 
    - **Remember*** to use the command responsibly and always review the summarized content to ensure its relevance and accuracy.
 
2. **!upload**: 
	- **Usage**: `!upload [attach a .pdf file]`
	- **Description**: When utilizing the `!upload` function, ensure you accompany it with an attached `.pdf` file. Upon executing this command with the appropriate attachment, the bot will commence the upload process. Post upload, it will meticulously extract all the textual content from the `.pdf` file. This extracted data is then systematically stored in a vector database for future retrievals and references. It's imperative to ensure the attached file is in `.pdf` format for the optimal functioning of this feature.
	- **Technical**:
		- `storing text data into a vector database`
			```python 
			def  vector_store(file_name, attachment):
				raw_documents  =  PyPDFLoader(attachment).load()
				text_splitter  =  CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
				docs  =  text_splitter.split_documents(raw_documents)
				return  docs
			```
		- `!upload command`
			```python
			@client.command(name='upload')
			async  def  upload(ctx):
				global  db
				if  not  ctx.message.attachments:
					await  ctx.send("Please attach a .pdf file with the `!upload` command.")	
				return
				
				attachment  =  ctx.message.attachments[0]
				attachment_url  =  attachment.url
				file_name  =  attachment.filename
				print(file_name)

				if  file_name.endswith('.pdf'):
					try:
						docs  =  vector_store(file_name, attachment_url)
						db  =  Chroma.from_documents(documents  =  docs, embedding  =  embeddings, persist_directory=  persist_directory)
						await  ctx.send("Text extracted and stored!")
					except  Exception  as  e:
						print(e)
						await  ctx.send("Failed to extract text!")
				else:
					await  ctx.send("Please upload a .pdf file!")
			```
3. **!chat**:
	- **Usage**: `!chat [ask a question]`
	- **Description**: Following the successful use of the `!upload` command, users unlock the capability to interact dynamically with their `.pdf` documents. By invoking the `!chat` command, you can initiate a conversation with a state-of-the-art Large Language Model (LLM). This allows for an engaging question-and-answer session with the content of your document. Simply pose your questions, and the LLM will delve into the extracted data from your document to provide informative answers. It's like having a dialogue with your document, powered by cutting-edge AI.
	- **Technical**: 
		- `!chat command`
		  ```python
		  @client.command(name='chat')
			async  def  chat(ctx, *, query: str):
				global  db
				try:
					retriever  =  db.as_retriever(search_type="mmr", search_kwargs={"k":1})
					qa  =  RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True, verbose  =  True)
					answer  =  qa({"query": query})
					strings  =  split_string(answer['result'], 2000)
					for  i  in  range(0, len(strings)):
						await  ctx.send(str(strings[i]))
				except  Exception  as  e:
					print(e)
					await  ctx.send("Use the !upload function to upload a .pdf or .txt file before chatting!")
		  ```
4. **!story**:
	- **Usage**: `!story [a short story title]`
	- **Description**: Engage your imagination with the `!story` command. By providing a short story title, you empower the bot to harness the capabilities of a sophisticated Large Language Model (LLM) to craft a unique short story inspired by that title. Each generated narrative is concise, ensuring it remains under 500 words, making it a perfect bite-sized read. But that's not all! Along with the text, the bot also conjures up a captivating image that resonates with the contents of the story, adding a visual dimension to your storytelling experience. Dive into a world where titles come to life, both in words and imagery.
	- **Technical**:
		- `function to generate short story`
			```python
			def  generate_story(story_idea):

				story  =  guidance('''
			{{#block hidden=True~}}
			You are a world class story teller; Your goal is to generate a interesting and captivating short story less than 500 words based on a story idea;

			Here is the story idea: {{story_idea}}
			Story: {{gen 'story' temperature=0}}
			{{/block~}}
			{{~story~}}
			''')

				image_url  =  guidance('''
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

				story  =  str(story(story_idea=story_idea))
				image_url  =  str(image_url(story  =  story))
				return  story, image_url
			```
		- `!story command`
			```python 
			@client.command(name='story')
			async  def  story(ctx, *, query: str):
				try:
					story, image_url  =  generate_story(query)
					strings  =  split_string(story, 2000)
					
					for  i  in  range(0, len(strings)):
						await  ctx.send(str(strings[i]))
					
					async  with  aiohttp.ClientSession() as  session:
						async  with  session.get(image_url) as  resp:
						if  resp.status  !=  200:
							return  await  ctx.send('Could not download image...')
						data  =  await  resp.read()
						image_stream  =  io.BytesIO(data)
						await  ctx.send(file=discord.File(fp=image_stream, filename="image.png"))
				except  Exception  as  e:
					print(e)
					await  ctx.send("Unable to generate a short story!")
			```

## Feedback and Contributions
We value your feedback! If you have any suggestions or have found bugs, please report them to email: calebkan1106@gmail.com or ryanfok8@gmail.com.

--- 

Stay tuned for regular updates and new features!
