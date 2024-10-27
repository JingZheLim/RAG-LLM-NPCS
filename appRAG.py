import cohere
import os
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from datetime import datetime
import json

# Simple RAG implementation, derived from the referenced github

game_name = 'Honkai_Star_Rail'
character_name = 'Acheron'

# Initializing Cohere client
co = cohere.Client('lMWcC83xVJlxEE5RahrziIdTiVqGetOp7Ba9YtD4')
print("Client's Initialised")

# Initializing embeddings
embeddings = CohereEmbeddings(cohere_api_key="lMWcC83xVJlxEE5RahrziIdTiVqGetOp7Ba9YtD4",
                              user_agent='langchain',
                              model="embed-english-v2.0")
print("Embedding's Initialised")

# Initializing Vector Stores
character_vectors = Chroma(persist_directory=f'{game_name}/characters/{character_name}/vectordb', embedding_function=embeddings)
world_vectors = Chroma(persist_directory=f'{game_name}/public_vectordb',embedding_function=embeddings)
print("Vector Database's Initialised")

# Initializing Chat History
current_session_filename = None
def initialize_chat_history():
    global current_session_filename
    
    # Create a directory of chat logs
    log_dir = "chat_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Make txt file with current time
    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_session_filename = os.path.join(log_dir, f"simpleRAG_{session_start}.txt")

    print("Chat history Initialised")

# Function to append chat history to the initialised txt file
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

# Retriving RAG content
def retrieve_relevant_content(query, k=1):
    # Search and retrieve relevant information from character and world database
    character_info = character_vectors.similarity_search(query,k=k)
    world_info = world_vectors.similarity_search(query,k=k)

    # Combine the informations for enhanced context
    character_context = "\n".join(doc.page_content for doc in character_info)
    world_context = "\n".join(doc.page_content for doc in world_info)

    # Combine both contexts
    combined = f"The following is for immersive purposes and is just background information \n Character Information: \n {character_context} \n\n World Information: \n {world_context}"

    return combined

def AIchatbot(query, chat_history):
    # Retrieve RAG context
    context = retrieve_relevant_content(query)

    # Prompt to sent to Cohere
    prompt = f"""
You are roleplaying as Acheron from Honkai Star Rail. Maintain her personality, speech patterns, and attitude consistently in all responses. Use the following context and chat history to inform your responses, but always stay in character.

{context}

Chat History:
"""
    # Get chat history and put into prompt
    for message in chat_history:
        prompt += f"Human: {message['human']}\nAcheron: {message['ai']}\n"
    prompt += f"Human: {query}\nAcheron:"

    # Generate response using generate from cohere
    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=300,
        temperature=0.6,  
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