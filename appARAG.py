import cohere
import os
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer, util
import torch

# Advaned RAG implementation, semantic search and reranking 

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

# Initializing SentenceTransformer for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer Initialised")

# Initializing Vector Stores
character_vectors = Chroma(persist_directory=f'{game_name}/characters/{character_name}/vectordb', embedding_function=embeddings)
world_vectors = Chroma(persist_directory=f'{game_name}/public_vectordb',embedding_function=embeddings)
print("Vector Database's Initialised")

# Initializing Chat History
current_session_filename = None
def initialize_chat_history():
    global current_session_filename
    
    log_dir = "chat_logs"
    os.makedirs(log_dir, exist_ok=True)

    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create file with identifiable name
    current_session_filename = os.path.join(log_dir, f"advancedRAG_{session_start}.txt")

    print("Chat history Initialised")

# Function to append chat history to the initialised txt file
def append_chat_history(message, is_human=True):
    global current_session_filename
    
    if current_session_filename is None:
        initialize_chat_history()

    with open(current_session_filename, "a", encoding="utf-8") as f:
        speaker = "Human" if is_human else "Acheron"
        json.dump({"speaker": speaker, "message": message}, f)
        f.write("\n")

def get_filename():
    return current_session_filename

# Real-time embedding and semantic search
def semantic_search(query, documents, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))
    
    return [documents[idx] for idx in top_results.indices]

# Retriving RAG content with semantic search and re-ranking
def retrieve_relevant_content(query, k=5):
    # Search and retrieve relevant information from character and world database
    character_docs = character_vectors.get()
    world_docs = world_vectors.get()
    
    character_results = semantic_search(query, character_docs['documents'], top_k=k)
    world_results = semantic_search(query, world_docs['documents'], top_k=k)
    
    # Combine and re-rank results
    all_results = character_results + world_results
    reranked_results = semantic_search(query, all_results, top_k=k)
    
    # Combine the information into one
    combined_context = "\n".join(reranked_results)
    
    return f"The following is for immersive purposes and is just background information:\n{combined_context}"

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