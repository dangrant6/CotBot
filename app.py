import os
import sys
from enum import Enum
from taipy.gui import Gui, State, notify, invoke_long_callback, navigate
import requests
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import tempfile
import pygame
import threading
import queue
import json
from typing import Any, List, Optional, Tuple
import time
from queue import Queue
import threading
from threading import Thread
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import re
qdrant_client = QdrantClient(host="localhost", port=6333)

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
audio_queue = queue.Queue()
# At the top of your file
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
transcription_queue = Queue()

# Add at the top of your file
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatMode(Enum):
    TEXT = "text"
    VOICE = "voice"

# Initial state variables
messages = [
    {"role": "system", "content": "You are a helpful, creative, clever, and very friendly AI assistant."},
    {"role": "user", "content": "Hello, who are you?"},
    {"role": "assistant", "content": "I am your new assistant! How can I help you today?"}
]
conversation = {
    "Conversation": ["Hi!", "Hi! I am Matey. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
is_recording = False
chat_mode = ChatMode.TEXT
recording_text = "Call"
mode_switch_text = "Voice"

# Add these to your initial state variables
is_listening = False
listening_timeout = 5  # seconds
energy_threshold = 4000  # Adjust based on testing
# After initializing recognizer
recognizer.energy_threshold = 300  # Lower threshold for better detection
recognizer.dynamic_energy_threshold = False  # Use fixed threshold
knowledge_base = None
kb_ready = False  # Add this line
kb_status = "Initializing Knowledge Base..." 

doc_keywords = {
    "Risk Adjustment": ["risk adjustment", "coding accuracy", "medical record retrieval", "interoperability"],
    "Low Income Subsidy": ["low income subsidy", "subsidy program", "eligibility"],
    "Onboarding Timeline": ["onboarding", "timeline", "new hire", "schedule"],
    "Welcome Letter": ["welcome", "new intern", "ace program", "red cell team", "communication", "expectations", "marco polo", "asana", "team communication", "methods of communication", "communicate", "communicating", "communication methods", "communication tools", "team expectations"],
    "First 10 Tasks": ["intern tasks", "first tasks", "to-do list", "initial responsibilities"]
}

# Threshold for similarity relevance to determine if a question is document-related
similarity_cutoff = 0.5

def init_qdrant_index(collection_name: str, vector_dim: int) -> str:
    # Check if the collection exists, and create it if not
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
    return collection_name

def add_documents_to_qdrant(collection_name: str, documents):
    # Assuming each document has a unique id and an embedding vector
    points = [
        PointStruct(id=i, vector=document.embedding, payload={"text": document.text})
        for i, document in enumerate(documents)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

class KnowledgeBase:
    def __init__(self):
        self.index = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the knowledge base with LlamaIndex"""
        try:
            # Initialize embedding model
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            # Load documents
            docs_path = Path("onboarding_docs")
            if not docs_path.exists():
                logger.error("onboarding_docs directory not found")
                return False
                
            documents = SimpleDirectoryReader(input_dir="onboarding_docs").load_data()
            if not documents:
                logger.error("No documents found in onboarding_docs directory")
                return False
            
            # Create the vector store index
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=Settings.embed_model
            )
            
            self.initialized = True
            logger.info("Knowledge base initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            return False
    
    def query(self, question: str) -> Tuple[str, List[str]]:
        """Query the knowledge base and return answer with sources"""
        try:
            if not self.initialized or not self.index:
                raise ValueError("Knowledge base not initialized")
                
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=3,  # Get top 3 most relevant chunks
                response_mode="compact"  # Get concise responses
            )
            
            # Get response
            response = query_engine.query(question)
            
            # Extract sources
            sources = []
            if hasattr(response, 'source_nodes'):
                sources = [
                    node.node.get_content()[:100] + "..."  # First 100 chars of each source
                    for node in response.source_nodes
                ]
            
            return str(response), sources
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return f"Error: {str(e)}", []

def check_kb_status(state: State):
    """Check knowledge base initialization status"""
    try:
        if not kb_manager.initialization_queue.empty():
            status, error = kb_manager.initialization_queue.get_nowait()
            if status == "success":
                notify(state, "success", "Knowledge base loaded successfully")
                state.kb_ready = True
            else:
                notify(state, "error", f"Knowledge base error: {error}")
                state.kb_ready = False
    except Exception as e:
        logger.error(f"Error checking KB status: {e}")

def init_kb() -> bool:
    """Initialize knowledge base and return status"""
    try:
        # Initialize embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Load documents
        docs_path = Path("onboarding_docs")
        if not docs_path.exists():
            logger.error("onboarding_docs directory not found")
            return False
            
        documents = SimpleDirectoryReader(input_dir="onboarding_docs").load_data()
        if not documents:
            logger.error("No documents found in onboarding_docs directory")
            return False
            
        # Create the vector store index directly and store on state
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=Settings.embed_model
        )
        
        # Create a query engine that will be reused
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        # Return both the index and query engine
        return True, (index, query_engine)
        
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        return False, None

def handle_kb_init_status(state: State, status: Any) -> None:
    """Handle knowledge base initialization status updates"""
    try:
        # Handle progress updates (integers)
        if isinstance(status, int):
            state.kb_status = f"Initializing Knowledge Base... ({status}%)"
            return
            
        # Handle final initialization result
        if isinstance(status, tuple) and len(status) == 2:
            success, kb_data = status
            
            if success and kb_data:
                index, query_engine = kb_data
                state.kb_index = index
                state.kb_query_engine = query_engine
                state.kb_ready = True
                state.kb_status = "Knowledge Base Ready"
                logger.info("Knowledge base successfully initialized and attached to state")
            else:
                state.kb_ready = False
                state.kb_status = "Failed to initialize knowledge base"
                logger.error("Knowledge base initialization failed")
                
        # Handle other status updates
        else:
            logger.warning(f"Received unexpected status type: {type(status)}")
            
    except Exception as e:
        logger.error(f"Error handling KB init status: {e}")
        state.kb_ready = False
        state.kb_status = f"Error: {str(e)}"


# Modify on_init to initialize knowledge base
def on_init(state: State) -> None:
    """Initialize application with proper state management and new knowledge base initialization"""
    try:
        # Basic state initialization
        state.messages = messages.copy()
        state.conversation = conversation.copy()
        state.current_user_message = current_user_message
        state.past_conversations = past_conversations.copy()
        state.selected_conv = selected_conv
        state.selected_row = selected_row.copy()
        state.is_recording = is_recording
        state.chat_mode = chat_mode
        state.recording_text = recording_text
        state.mode_switch_text = mode_switch_text
        state.is_listening = is_listening
        
        # Knowledge base initialization
        state.kb_ready = False
        state.kb_status = "Initializing Knowledge Base..."
        state.knowledge_base = None

        logger.info("Starting knowledge base initialization...")
        invoke_long_callback(
            state,
            init_kb,  # This is our new init_kb function
            [],
            handle_kb_init_status,  # This is our new status handler
            period=1000
        )
        
        logger.info("Application initialization completed")
        
    except Exception as e:
        logger.error(f"Error in on_init: {e}")
        state.kb_status = f"Error during initialization: {str(e)}"
        state.kb_ready = False
        notify(state, "error", "Failed to initialize application")
        
def on_input_change(state: State, var_name: str, var_value: str) -> None:
    """Handle input changes"""
    # You can add input processing logic here if needed
    pass

def is_relevant_to_documents(question: str) -> bool:
    """Enhanced document relevance checking with better logging"""
    question_lower = question.lower()
    
    # Log the incoming question for debugging
    logger.info(f"Checking document relevance for: {question}")
    
    for category, keywords in doc_keywords.items():
        for keyword in keywords:
            if keyword.lower() in question_lower:
                logger.info(f"Match found in category '{category}' with keyword '{keyword}'")
                return True
                
            # Add fuzzy matching for similar terms
            if any(term in keyword.lower() for term in ["communication", "expectations", "methods"] 
                  if term in question_lower):
                logger.info(f"Fuzzy match found in category '{category}'")
                return True
    
    logger.info("No relevant document matches found")
    return False

def request(state: State, prompt: str) -> str:
    """Enhanced request function with better KB integration"""
    try:
        logger.info(f"Processing request: {prompt}")
        
        # Check if KB is ready and question is relevant
        if state.kb_ready and hasattr(state, 'kb_query_engine') and is_relevant_to_documents(prompt):
            logger.info("Using knowledge base for response")
            try:
                # Use the stored query engine
                response = state.kb_query_engine.query(prompt)
                
                # Extract sources if available
                sources = []
                if hasattr(response, 'source_nodes'):
                    sources = [
                        node.node.get_content()[:100] + "..."
                        for node in response.source_nodes
                    ]
                
                # Format response
                return format_kb_response(str(response), sources)
                
            except Exception as kb_error:
                logger.error(f"Knowledge base query failed: {kb_error}")
        
        # Fall back to conversational API
        logger.info("Using conversational API")
        return original_request(state, prompt)
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        return error_msg

def original_request(state: State, prompt: str) -> str:
    """Handle general conversational responses"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return "I apologize, but I'm having trouble connecting to my API. Could you please check the API key configuration?"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:5000",
            "Content-Type": "application/json"
        }

        data = {
            "model": "google/gemma-2-9b-it:free",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Matey, a friendly and knowledgeable assistant. "
                        "Keep responses natural and conversational while being "
                        "informative. Engage with users as if chatting with a colleague. "
                        "For technical topics, be clear but maintain a warm tone. Do not use any asterisks or emojis."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            state.messages.append({"role": "assistant", "content": content})
            return content
        else:
            return "I'm having trouble processing your request. Could you try rephrasing it?"
            
    except Exception as e:
        return f"I apologize, but I ran into an error. How else can I help you?"

def format_kb_response(answer: str, sources: List[str]) -> str:
    """Format the knowledge base response with sources"""
    response_parts = [answer.strip()]
    
    if sources:
        response_parts.append("\nRelevant sources:")
        for idx, source in enumerate(sources, 1):
            response_parts.append(f"{idx}. {source}")
            
    return "\n\n".join(response_parts)

def send_message(state: State) -> None:
    """Enhanced send_message with more debugging"""
    print(f"DEBUG: send_message called with message: {state.current_user_message}")
    
    try:
        if not state.current_user_message.strip():
            print("DEBUG: Empty message received")
            return
            
        message_to_send = state.current_user_message.strip()
        print(f"DEBUG: Sending to API: {message_to_send}")
        
        # Get API response
        answer = request(state, message_to_send)
        print(f"DEBUG: API response: {answer}")
        
        if answer and not answer.startswith(("Error:", "API Error:", "Network error")):
            # Update conversation
            new_conv = {
                "Conversation": state.conversation["Conversation"] + [message_to_send, answer]
            }
            state.conversation = new_conv
            state.current_user_message = ""
            state.selected_row = [len(new_conv["Conversation"])]
            
            # Handle voice response
            if state.chat_mode == ChatMode.VOICE:
                print("DEBUG: Starting text-to-speech")
                threading.Thread(
                    target=speak_response,
                    args=(answer,),
                    daemon=True
                ).start()
                print("DEBUG: Text-to-speech thread started")
            
            notify(state, "success", "Response received")
        else:
            print(f"DEBUG: Invalid API response: {answer}")
            notify(state, "error", "Failed to get valid response")
            
    except Exception as e:
        print(f"DEBUG: Error in send_message: {e}")
        logger.error(f"Error in send_message: {e}")


def voice_recording_task() -> bool:
    """Modified to use queue for passing text"""
    print("DEBUG: Starting voice_recording_task")
    try:
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info("Listening for speech...")
            audio = recognizer.listen(source)
            
            logger.info("Speech detected, transcribing...")
            text = recognizer.recognize_google(audio)
            logger.info(f"Transcribed: {text}")
            
            # Put the transcribed text in the queue
            print(f"DEBUG: Putting text in queue: {text}")
            transcription_queue.put(text)
            return True
            
    except Exception as e:
        logger.error(f"Error in recording: {e}")
        transcription_queue.put(f"Error: {str(e)}")
        return False

def voice_recording_status(state: State, status: Any) -> None:
    """Enhanced status handler with better logging"""
    try:
        logger.info(f"Status type: {type(status)}, Value: {status}")
        
        if isinstance(status, str):
            logger.info(f"Processing transcribed text: {status}")
            
            # Set the message
            state.current_user_message = status
            logger.info("Updated current_user_message")
            
            # Ensure voice mode is set
            state.chat_mode = ChatMode.VOICE
            logger.info("Ensured voice mode is set")
            
            # Send the message
            logger.info("About to call send_message...")
            send_message(state)
            
        elif isinstance(status, bool):
            logger.info("Received boolean status - ignoring")
            # Don't do anything for boolean status
            pass
            
        elif isinstance(status, Exception):
            logger.error(f"Voice recording error: {str(status)}")
            notify(state, "error", str(status))
            
        else:
            logger.warning(f"Unexpected status type: {type(status)}")
            
        # Always reset recording state after processing
        state.is_recording = False
        state.recording_text = "Start Recording"
        
    except Exception as e:
        logger.error(f"Error in voice recording status: {e}", exc_info=True)
        notify(state, "error", f"Error processing voice input: {str(e)}")
        state.is_recording = False
        state.recording_text = "Start Recording"

def speak_response(text: str) -> None:
    """Enhanced text-to-speech with immediate feedback"""
    print(f"DEBUG: speak_response called with: {text}")  # Direct print
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            print(f"DEBUG: Created temp file: {temp_filename}")  # Direct print
            
        # Generate speech
        print("DEBUG: Generating speech...")  # Direct print
        tts = gTTS(text=text, lang='en')
        tts.save(temp_filename)
        print("DEBUG: Speech file saved")  # Direct print
        
        # Play audio
        print("DEBUG: Playing audio...")  # Direct print
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for playback
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        print("DEBUG: Audio playback completed")  # Direct print
        
    except Exception as e:
        print(f"DEBUG: Error in speak_response: {e}")  # Direct print
        logger.error(f"Error in speak_response: {e}", exc_info=True)

def toggle_recording(state: State) -> None:
    """Toggle voice recording on and off, navigating to the call page."""
    print("DEBUG: toggle_recording called")
    try:
        if state.is_recording:
            print("DEBUG: Stopping recording")
            state.is_recording = False
            state.recording_text = "Start Recording"
            notify(state, "info", "Recording stopped")
            navigate(state, "main")
        else:
            print("DEBUG: Starting recording")
            state.is_recording = True
            state.recording_text = "Listening..."
            notify(state, "info", "Listening for voice input...")
            
            # Navigate to call page before starting recording
            navigate(state, "call")
            
            # Start the recording task after navigation
            invoke_long_callback(
                state,
                voice_recording_task,
                [],
                handle_voice_result,
                period=100
            )
    except Exception as e:
        print(f"DEBUG: Error in toggle_recording: {e}")
        logger.error(f"Error in toggle_recording: {e}")
        state.is_recording = False
        state.recording_text = "Start Recording"
        navigate(state, "main")

def end_call(state: State) -> None:
    """End the call and return to the main chat page."""
    try:
        print("DEBUG: Ending call")
        # Stop recording first
        state.is_recording = False
        state.recording_text = "Start Recording"
        
        # Clear any pending audio
        try:
            pygame.mixer.music.stop()
        except:
            pass
            
        # Navigate back to main page
        navigate(state, "main")
        notify(state, "info", "Call ended")
        
    except Exception as e:
        print(f"DEBUG: Error in end_call: {e}")
        logger.error(f"Error in end_call: {e}")
        navigate(state, "main")  # Ensure we return to main page even if there's an error

        
def handle_voice_result(state: State, result: Any) -> None:
    """Modified handler to check queue"""
    print(f"DEBUG: handle_voice_result received: {result}")
    
    try:
        # Process any available transcription
        process_transcription(state)
        
        # Reset recording state if done
        if isinstance(result, bool) and result:
            state.is_recording = False
            state.recording_text = "Start Recording"
            
    except Exception as e:
        print(f"DEBUG: Error in handle_voice_result: {e}")
        logger.error(f"Error in handle_voice_result: {e}")
        state.is_recording = False
        state.recording_text = "Start Recording"
        
def process_transcription(state: State) -> None:
    """Process transcribed text from queue"""
    try:
        # Get text from queue without blocking
        if not transcription_queue.empty():
            text = transcription_queue.get_nowait()
            print(f"DEBUG: Got text from queue: {text}")
            
            if text.startswith("Error:"):
                print(f"DEBUG: Error in transcription: {text}")
                notify(state, "error", text)
                return
                
            # Process the transcribed text
            state.current_user_message = text
            state.chat_mode = ChatMode.VOICE
            
            # Send message
            print("DEBUG: Calling send_message with transcribed text")
            send_message(state)
            
    except Exception as e:
        print(f"DEBUG: Error processing transcription: {e}")
        logger.error(f"Error processing transcription: {e}")

def switch_mode(state: State) -> None:
    """Switch between text and voice chat modes"""
    try:
        state.chat_mode = ChatMode.VOICE if state.chat_mode == ChatMode.TEXT else ChatMode.TEXT
        state.is_recording = False
        state.recording_text = "Start Recording"
        state.mode_switch_text = "Voice" if state.chat_mode == ChatMode.TEXT else "Text"
        notify(state, "info", f"Switched to {state.chat_mode.value} mode")
    except Exception as e:
        notify(state, "error", f"Error switching modes: {str(e)}")

def style_conv(state: State, idx: int, row: int) -> str:
    """Apply style to conversation messages based on sender."""
    if idx is None:
        return None
    return "user_message" if idx % 2 == 0 else "gpt_message"

def reset_chat(state: State) -> None:
    """Reset the chat conversation"""
    try:
        state.past_conversations.append([len(state.past_conversations), state.conversation])
        state.conversation = conversation.copy()
        state.messages = messages.copy()
        state.selected_row = [1]
    except Exception as e:
        notify(state, "error", f"Error resetting chat: {str(e)}")

def tree_adapter(item: list) -> tuple[str, str]:
    """Convert past conversation to tree item"""
    try:
        identifier = item[0]
        if len(item[1]["Conversation"]) > 3:
            return (identifier, item[1]["Conversation"][2][:50] + "...")
        return (identifier, "Empty conversation")
    except Exception:
        return (0, "Error loading conversation")

def select_conv(state: State, var_name: str, value) -> None:
    """Select a past conversation"""
    try:
        state.conversation = state.past_conversations[value[0][0]][1]
        state.messages = messages.copy()
        for i in range(2, len(state.conversation["Conversation"]), 2):
            user_msg = state.conversation["Conversation"][i]
            assistant_msg = state.conversation["Conversation"][i + 1]
            state.messages.append({"role": "user", "content": user_msg})
            state.messages.append({"role": "assistant", "content": assistant_msg})
        state.selected_row = [len(state.conversation["Conversation"])]
    except Exception as e:
        notify(state, "error", f"Error selecting conversation: {str(e)}")
        
main_page_md = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Matey **- R**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|on_action=reset_chat|>
<|{kb_status}|text|class_name=kb-status|>
### Previous chats ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|row_class_name=style_conv|show_all|selected={selected_row}|rebuild|hover=True|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|on_change=on_input_change|>
<|part|class_name=voice-controls|visible={chat_mode == ChatMode.VOICE}|
<|{recording_text}|button|class_name=record-button|on_action=toggle_recording|>
<|{current_user_message}|text|class_name=voice-transcript|>
|>
|>
|>
|>
"""

call_page_md = """
<|part|class_name=call-page|
<|part|class_name=avatar-container|
<|part|class_name=avatar-pulse|>|>
<|part|class_name=avatar-image|
<|{True}|image|content=headshot.png|width=150px|height=150px|>
|>
|>
<|Recording...|text|class_name=recording-status|>
<|Hang Up|button|class_name=hang-up-button|on_action=end_call|>
|>
"""

pages = {
    "main": main_page_md,
    "call": call_page_md
}

if __name__ == "__main__":
    load_dotenv()
    try:
        import os
        # Get absolute path to the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create path mapping for static files
        path_mapping = {
            ".": current_dir,
            "images": current_dir
        }
        
        stylekit = {
            "color_background_dark": "#1e1e1e"
        }
        
        gui = Gui(pages=pages, path_mapping=path_mapping)
        gui.run(title="💬 VCell-R", dark_mode=True, stylekit=stylekit)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
