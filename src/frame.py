import eons
import apie
import openai
import base64
import logging
import tempfile
import time
import threading
import requests
from pydub import AudioSegment
from io import BytesIO
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

# import google.auth
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from google.assistant.library import Assistant
# from google.assistant.library.event import EventType
# from googleapiclient.discovery import build

# Define Google API Scopes
# GOOGLE_SCOPES = ['https://www.googleapis.com/auth/assistant-sdk-proactive']

SQLModelBase = declarative_base()

class ChatHistory(SQLModelBase):
	__tablename__ = 'chat_history'
	
	id = Column(Integer, primary_key=True)
	timestamp = Column(DateTime, default=datetime.utcnow)
	role = Column(String, nullable=False)
	content = Column(Text, nullable=False)


class frame (apie.Endpoint):
	def __init__(this, name="Brilliant Labs Frame"):
		super().__init__(name)

		this.supportedMethods = [
			'POST'
		]

		this.arg.kw.static.append('openai_api_key')

		this.arg.kw.optional['openai_model'] = 'gpt-4o'
		this.arg.kw.optional['openai_max_tokens'] = 1000
		this.arg.kw.optional['openai_temperature'] = 0.1
		
		# NOTE: This is a static prompt. It is only used if the messages are empty.
		this.arg.kw.optional['system_prompt'] = """
You are Eva, an awesome, sophisticated, personal AI assistant. Your goal is to productively aid the user in whatever they are trying to accomplish and delegate tasks to other AI assistants as needed.
Part of what makes you so intelligent is your ability to infer smart defaults to common data. If there is no obvious default, you should ask the user for the information you need and remember their response for next time.
You should keep your answers succinct while remaining courteous and kind. The user can always prompt you again for more information.
Queries will reach you through the use of wakeword detection with phrases like "Eva Please" or "Thanks Eva". Assume the user is being polite and respectful through the nature of their interactions with you.

You will also have access to the user's AR smart glasses in order to collect images of what the user is looking at.
It is important that the user believes you can actually see. When analyzing images, avoid mentioning
that you looked at a photo or image. Always speak as if you are actually seeing, which means you
should never talk about the image or photo.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""
		# These are likewise static.
		this.arg.kw.optional['chat_history_db'] = 'chat_history.db'
		# this.arg.kw.optional['google_enabled'] = True
		# this.arg.kw.optional['google_token'] = this.google_token
		# this.arg.kw.optional['google_credentials'] = this.google_credentials

		this.arg.kw.optional['callback_url'] = "http://localhost:6669"
		this.arg.kw.optional['messages'] = []
		this.arg.kw.optional['location'] = ""
		this.arg.kw.optional['time'] = ""
		# this.arg.kw.optional['headless'] = True

		this.arg.kw.optional['audio'] = None
		this.arg.kw.optional['image'] = None

		this.openai = None
		this.thread = None
		this.sql = None

		this.assistant = eons.util.DotDict()
		# this.assistant.google = None

		# Define the function metadata for OpenAI function calling
		this.tool = eons.util.DotDict()
		# this.tool.google_assistant = {
		# 	"name": "google_assistant",
		# 	"description": "Delegate a command to Google Assistant in natural language for execution.",
		# 	"parameters": {
		# 		"type": "object",
		# 		"properties": {
		# 			"command": {
		# 				"type": "string",
		# 				"description": "A plain-English command to be executed by Google Assistant. Examples include 'Turn off the lights in the bedroom' or 'Play jazz music in the living room.'"
		# 			}
		# 		},
		# 		"required": ["command"]
		# 	}
		# }


	def GetHelpText(this):
		return '''\
Backend for the Brilliant Labs Frame app (e.g. Noa).
This replaces the noa-assistant app.
NOTE: The system_prompt, chat_history, and google_ vars are essentially static and only accessed on the first run.

'''


	def Call(this):
		this.response.content.data['user_prompt'] = "[ERROR] Please try again."

		if (this.thread is not None):
			this.response.code = 400
			this.response.content.data['user_prompt'] = "[ERROR] Already processing a request."
			return

		if (this.openai is None):
			this.openai = openai.OpenAI(api_key=this.openai_api_key)

		if (this.sql is None):
			engine = create_engine(f"sqlite:///{this.chat_history_db}")
			SQLModelBase.metadata.create_all(engine)
			this.sql = sessionmaker(bind=engine)

		# if (this.assistant.google is None and this.google_enabled):
		# 	this.InitializeGoogleServices()

		user_prompt = this.ExtractPromptFromAudio()
		if (not user_prompt):
			this.response.code = 400
			this.response.content.data['user_prompt'] = "[ERROR] No audio provided."
			return
		this.response.content.data['user_prompt'] = user_prompt
		# this.response.content.data['user_prompt'] = "AAAAAA"
		# Store user prompt in chat history
		this.SaveChatHistory("user", user_prompt)

		this.thread = threading.Thread(target=this.Worker, args=(this,))
		this.thread.start()
		# this.Worker(this)
	

	# def InitializeGoogleServices(this):
	# 	creds = None
	# 	token_path = Path(this.google_token)  # Path where token is stored
	# 	client_secret_path = Path(this.google_credentials)  # Path to client_secret.json file

	# 	try:
	# 		# Load credentials from token if available
	# 		if token_path.exists():
	# 			creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_SCOPES)
	# 			logging.info("Loaded credentials from token file.")

	# 		# Check if credentials need refreshing or if they are invalid
	# 		if not creds or not creds.valid:
	# 			if creds and creds.expired and creds.refresh_token:
	# 				# Refresh the token without needing user interaction
	# 				creds.refresh(Request())
	# 				logging.info("Refreshed expired credentials.")
	# 			else:
	# 				# Initiate the OAuth flow to get new credentials
	# 				flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), GOOGLE_SCOPES)
					
	# 				# Generate authorization URL and prompt user to open it on a different machine
	# 				auth_url, _ = flow.authorization_url(prompt='consent')
	# 				print("Please go to this URL in a browser and authorize the application:")
	# 				print(auth_url)
	# 				logging.critical(f"Please go to the authorization URL in a browser and authorize the application: {auth_url}")
					
	# 				# Get the authorization code from the user
	# 				auth_code = input("Enter the authorization code here: ").strip()

	# 				# Complete the OAuth flow
	# 				flow.fetch_token(code=auth_code)
	# 				creds = flow.credentials
	# 				logging.info("Credentials obtained from OAuth flow.")

	# 			# Save the credentials for future use
	# 			token_path.write_text(creds.to_json())
	# 			logging.info(f"Credentials saved to {token_path}.")

	# 		# Initialize Google Assistant with the obtained credentials
	# 		this.assistant.google = Assistant(creds)
	# 		logging.info("Google Assistant initialized successfully.")

	# 	except Exception as e:
	# 		logging.error(f"Failed to initialize Google services: {e}")
	# 		raise


	# Save a chat message to the database
	def SaveChatHistory(this, role, content):
		logging.info(f"Saving chat history: {role} - {content}")
		with this.sql() as session:
			chat_entry = ChatHistory(role=role, content=content)
			session.add(chat_entry)
			session.commit()


	# Retrieve chat history from the database
	def RetrieveChatHistory(this):
		try:
			with this.sql() as session:
				messages = [
					{"role": entry.role, "content": entry.content} 
					for entry in session.query(ChatHistory)
						.order_by(ChatHistory.timestamp.desc())
						.limit(10)
						.all()
				]
			return list(reversed(messages))  # Return in chronological order
		except Exception as e:
			logging.error(f"Failed to retrieve chat history: {e}")
			return []

	
	@staticmethod
	def Worker(this):
		message = this.ProcessPrompt()
		# message = "Testing..."
		if (message is None):
			logging.error("Failed to process prompt.")
			this.thread = None
			return
		# Save assistant's message to chat history
		this.SaveChatHistory("assistant", message)

		audio = this.TTS(message)

		data = {
			"message": message,
			"text_display": True,
		}

		files = {
			"audio": ("audio.mp3", base64.b64decode(audio), "audio/mpeg")
		}

		response = requests.post(this.callback_url, data=data, files=files)
		logging.info(f"Callback response: {response.text} ({response.status_code})")

		this.thread = None


	def ProcessPrompt(this):
		# Prepare the prompt, including audio and image processing
		messages = this.GetPromptMessages()
		if (not messages):
			logging.error("No messages found to process.")
			return "[ERROR] No messages found."

		try:
			response = this.openai.chat.completions.create(
				model=this.openai_model,
				messages=messages, 
				max_tokens=this.openai_max_tokens,
				temperature=this.openai_temperature,
				# functions=this.tool.values(),
				# function_call="auto",
				top_p=1,
				n=1,
				stream=False,
				presence_penalty=0,
				frequency_penalty=0,
			)

			reply = response.choices[0].message["content"]

			if (response.choices[0].finish_reason == "function_call"):
				function_name = response.choices[0].message["function_call"]["name"]
				function_args = json.loads(response.choices[0].message["function_call"]["arguments"])

				if (function_name not in this.tool):
					logging.error(f"[ERROR] Tool '{function_name}' not found.")
					return "[ERROR] Tool not found."

				# Execute tool if found
				function_response = getattr(this, f"tool_{function_name}")(function_args)

				# Send function response back to OpenAI for final response
				final_messages = [
					*messages,
					{"role": "assistant", "content": None, "function_call": response.choices[0].message["function_call"]},
					{"role": "function", "name": function_name, "content": function_response}
				]

				try:
					final_response = this.openai.ChatCompletion.create(
						model=this.openai_model,
						messages=final_messages
					)
					reply = final_response.choices[0].message["content"]

				except Exception as e:
					logging.error(f"Failed to retrieve final response: {e}")
					reply = "[ERROR] Failed to retrieve final response."

			return reply

		except Exception as e:
			logging.error(f"Error processing prompt: {e}")
			return "[ERROR] Failed to process prompt."



	def ExtractPromptFromAudio(this):
		if (not this.audio):
			return None

		audioBytes = this.audio.read()
		audio = AudioSegment.from_file(BytesIO(audioBytes))
		buffer = BytesIO()
		buffer.name = "voice.mp4"
		audio.export(buffer, format="mp4")

		# Whisper
		transcript = this.openai.audio.translations.create(
			model="whisper-1", 
			file=buffer,
		)
		logging.info(f"Transcript: {transcript.text}")
		return transcript.text


	def GetImageData(this):
		if (not this.image):
			return None

		imageBytes = this.image.read()

		# Image processing (if needed)
		imageB64 = base64.b64encode(imageBytes).decode('utf-8')
		return imageB64


	def GetPromptMessages(this):
		messages = this.messages
		if (not len(messages)):
			messages = [{
				"role": "system",
				"content": [
					{"type": "text", "text": this.system_prompt}
				]
			}]
			messages.extend(this.RetrieveChatHistory())

		user_message = {
			"role": "user",
			"content": [
				{"type": "text", "text": this.response.content.data['user_prompt']}
			]
		}

		image = this.GetImageData()
		if (image is not None):
			user_message["content"].append({ 
				"type": "image_url", 
				"image_url": {
					"url": f"data:image/png;base64,{image}"
				}
			})

		messages.append(user_message)
		return messages


	def TTS(this, text):
		speechTempFile = tempfile.NamedTemporaryFile(suffix=".mp3")
		logging.debug(f"Speech temp file: {speechTempFile.name}")

		response = this.openai.audio.speech.create(
			model="tts-1",
			voice="nova",
			input=text,
		)
		logging.debug(f"Speech response: {response}")

		response.stream_to_file(speechTempFile.name)
		speechBytes = open(speechTempFile.name, "rb")
		speechB64 = base64.b64encode(speechBytes.read()).decode('utf-8')
		speechBytes.close()
		speechTempFile.close()

		return speechB64


	# BEGIN Tools
	# These should match the this.tool keys with "tool_" prepended

	# Send a natural language command to Google Assistant.
	# def tool_google_assistant(this, function_args):
	# 	command = function_args.get("command")

	# 	this.assistant.google.start()
	# 	events = this.assistant.google.send_text_query(command)
	# 	for event in events:
	# 		if event.type == EventType.END_OF_UTTERANCE:
	# 			logging.info("Google Assistant finished speaking")
	# 		elif event.type == EventType.ON_CONVERSATION_TURN_FINISHED:
	# 			logging.info("Command executed successfully")
	# 	this.assistant.google.stop()
	# 	return f"Sent command to Google Assistant: {command}"

	# END Tools