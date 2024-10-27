import cohere
import os
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from datetime import datetime
import json

# Simplest Implementation, NO RAG

game_name = 'Honkai_Star_Rail'
character_name = 'Acheron'

# Initializing Cohere client
co = cohere.Client('lMWcC83xVJlxEE5RahrziIdTiVqGetOp7Ba9YtD4')
print("Client's Initialised")

# Initializing Chat History
current_session_filename = None
def initialize_chat_history():
    global current_session_filename
    
    # Create a directory of chat logs
    log_dir = "chat_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Make txt file with current time
    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_session_filename = os.path.join(log_dir, f"chat_history_{session_start}.txt")

    print("Chat history Initialised")

def append_chat_history(message, is_human=True):
    global current_session_filename
    
    if current_session_filename is None:
        initialize_chat_history()

    # Append the message to the file
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(current_session_filename, "a", encoding="utf-8") as f:
        speaker = "Human" if is_human else "Acheron"
        # json.dump({"timestamp": timestamp, "speaker": speaker, "message": message}, f)
        json.dump({"speaker": speaker, "message": message}, f)
        f.write("\n")

def get_filename():
    return current_session_filename

def AIchatbot(query, chat_history):
    # Prepare prompt to send to Cohere
    prompt = f"""
You are roleplaying as Acheron from Honkai Star Rail. Maintain her personality, speech patterns, and attitude consistently in all responses. Use the following context and chat history to inform your responses, but always stay in character.

Chat History:
"""
    for message in chat_history:
        prompt += f"Human: {message['human']}\nAcheron: {message['ai']}\n"
    prompt += f"Human: {query}\nAcheron:"

    # Generate response using Cohere
    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=300,
        temperature=0.6,  # Slightly reduced for more consistent character portrayal
        k=0,
        stop_sequences=["Human:", "\n"],
        return_likelihoods='NONE'
    )
    
    return response.generations[0].text.strip()


initialize_chat_history()

# Main conversation loop
chat_history = []
initial_message = "*A figure shrouded in darkness emerges, her piercing gaze fixed upon you* What brings you to seek me out, mortal?"
print(f"Acheron: {initial_message}")
append_chat_history(initial_message, is_human=False)

while True:
    user_input = input("You: ")
    append_chat_history(user_input, is_human=True)
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        farewell_message = "Farewell, mortal. Until our paths cross again."
        print(f"Acheron: {farewell_message}")
        append_chat_history(farewell_message, is_human=False)
        break
    
    response = AIchatbot(user_input, chat_history)
    print(f"Acheron: {response}")
    append_chat_history(response, is_human=False)
        
    chat_history.append({"human": user_input, "ai": response})