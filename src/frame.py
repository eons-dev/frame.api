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
You are Yuno, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

It is important that the user believes you can actually see. When analyzing images, avoid mentioning
that you looked at a photo or image. Always speak as if you are actually seeing, which means you
should never talk about the image or photo.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""

		this.arg.kw.optional['callback_url'] = "http://localhost:6669"
		this.arg.kw.optional['messages'] = []
		this.arg.kw.optional['location'] = ""
		this.arg.kw.optional['time'] = ""

		this.arg.kw.optional['audio'] = None
		this.arg.kw.optional['image'] = None

		this.openai = None
		this.thread = None


	def GetHelpText(this):
		return '''\
Backend for the Brilliant Labs Frame app (e.g. Noa).
This replaces the noa-assistant app.
NOTE: The system_prompt is essentially static. It is only used if the messages are empty.
'''


	def Call(this):
		this.response.content.data['user_prompt'] = "[ERROR] Please try again."
		
		# TODO: What is this?
		# this.response.content.data['debug'] = {
		# 	"topic_changed": False
		# }

		if (this.thread is not None):
			this.response.code = 400
			this.response.content.data['user_prompt'] = "[ERROR] Already processing a request."
			return

		if (this.openai is None):
			this.openai = openai.OpenAI(api_key=this.openai_api_key)

		# user_prompt = this.ExtractPromptFromAudio()
		# if (not user_prompt):
		# 	this.response.code = 400
		# 	this.response.content.data['user_prompt'] = "[ERROR] No audio provided."
		# 	return
		# this.response.content.data['user_prompt'] = user_prompt
		this.response.content.data['user_prompt'] = "AAAAAA"

		this.Worker(this)

		# this.thread = threading.Thread(target=this.Worker, args=(this,))
		# this.thread.start()
	

	@staticmethod
	def Worker(this):
		# message = this.ProcessPrompt()
		message = "Testing..."
		if (message is None):
			logging.error("Failed to process prompt.")
			this.thread = None
			return

		audioBytes = this.audio.read()

		# Create a file-like object for Whisper API to consume
		audio = AudioSegment.from_file(BytesIO(audioBytes))
		buffer = BytesIO()
		buffer.name = "voice.mp3"
		audio.export(buffer, format="mp3")

		# audio = this.TTS(message)

		data = {
			"message": message,
			"text_display": True,
		}

		files = {
			# "audio": ("audio.mp3", base64.b64decode(audio), "audio/mpeg")
			"audio": ("audio.mp3",buffer, "audio/mpeg")
		}

		response = requests.post(this.callback_url, data=data, files=files)
		logging.info(f"Callback response: {response.text} ({response.status_code})")

		this.thread = None


	def ProcessPrompt(this):
		# Prepare the prompt.
		# This includes audio and image processing.
		messages = this.GetPromptMessages()
		if (messages is None):
			return

		response = this.openai.chat.completions.create(
			model=this.openai_model,
			messages=messages,  # The chat history
			max_tokens=this.openai_max_tokens,
			temperature=this.openai_temperature,
			top_p=1,
			n=1,
			stream=False,
			presence_penalty=0,
			frequency_penalty=0,
		)

		return response.choices[0].message.content


	def ExtractPromptFromAudio(this):
		if (not this.audio):
			return None

		audioBytes = this.audio.read()

		# Create a file-like object for Whisper API to consume
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

		# TODO: Image processing, if needed.

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
			voice="alloy",
			input=text,
			speed=1.3,
		)
		logging.debug(f"Speech response: {response}")

		response.stream_to_file(speechTempFile.name)
		speechBytes = open(speechTempFile.name, "rb")
		speechB64 = base64.b64encode(speechBytes.read()).decode('utf-8')
		speechBytes.close()
		speechTempFile.close()

		return speechB64