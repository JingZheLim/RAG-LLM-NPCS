import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from datetime import datetime

# Gets and loads the chat history file
def load_chat_history(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Gets and loads the reference responses file
def load_reference_responses(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Calculates the BLEU scores 
def calculate_bleu_scores(chat_history, reference_responses):
    bleu_scores = []
    current_question = None

    # Go through the chat history file
    for entry in chat_history:
        # if the entry of speaker is human then make the key (the question) as the current key
        if entry['speaker'] == 'Human':
            current_question = entry['message']
            # else check if it is the character speaking and find the matching reference
        elif entry['speaker'] == 'Acheron' and current_question:
            candidate = word_tokenize(entry['message'].lower())
            
            # Find matching reference responses for the current question
            matching_references = [word_tokenize(ref.lower()) for ref in reference_responses.get(current_question, [])]
            
            # Once matched calculated the BLEU score
            if matching_references:
                score = sentence_bleu(matching_references, candidate)
                bleu_scores.append((current_question, entry['message'], score))
            else:
                bleu_scores.append((current_question, entry['message'], None))

    return bleu_scores

# Calculates the average BLEU score
def calculate_average_bleu_score(bleu_scores):
    valid_scores = [score for _, _, score in bleu_scores if score is not None]
    if valid_scores:
        return sum(valid_scores) / len(valid_scores)
    else:
        return None

# Writes the results to a specified file
def write_results_to_file(bleu_scores, average_score, output_file):
    with open(output_file, 'w') as f:
        f.write("BLEU Scores for Acheron's responses:\n")
        for question, response, score in bleu_scores:
            f.write(f"Question: {question}\n")
            f.write(f"Response: {response}\n")
            if score is not None:
                f.write(f"BLEU Score: {score:.4f}\n")
            else:
                f.write("BLEU Score: No reference available\n")
            f.write("-" * 50 + "\n")

        if average_score is not None:
            f.write(f"\nAverage BLEU Score: {average_score:.4f}\n")
        else:
            f.write("\nAverage BLEU Score: Could not be calculated (no valid scores)\n")

# File paths for easier testing
chat_file = 'chat_logs/simpleRAG_20241024_192740.txt'
reference_file = 'reference.json'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'simple_bleu_scores_{timestamp}.txt'

# Run all of the functions 
chat_history = load_chat_history(chat_file)
reference_responses = load_reference_responses(reference_file)
bleu_scores = calculate_bleu_scores(chat_history, reference_responses)
average_score = calculate_average_bleu_score(bleu_scores)

# Print out the results into a text document
write_results_to_file(bleu_scores, average_score, output_file)

# Also print results to console
print("BLEU Scores for Acheron's responses:")
for question, response, score in bleu_scores:
    print(f"Question: {question}")
    print(f"Response: {response}")
    if score is not None:
        print(f"BLEU Score: {score:.4f}")
    else:
        print("BLEU Score: No reference available")
    print("-" * 50)

if average_score is not None:
    print(f"\nAverage BLEU Score: {average_score:.4f}")
else:
    print("\nAverage BLEU Score: Could not be calculated (no valid scores)")
