from flask import Flask, request, jsonify, render_template
from pythainlp.tokenize import word_tokenize
from torchtext.data.utils import get_tokenizer
import torch, torchdata, torchtext
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random, math, time
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
from Seq2SeqTransformer import select_model
import pickle
token_transform = {}
vocab_transform = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocab_transform from pickle file
with open('vocab_transform.pkl', 'rb') as f:
    vocab_transform = pickle.load(f)

# Define the Thai tokenizer function
def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

token_transform = {
    "en": get_tokenizer('spacy', language='en_core_web_sm'),
    "th": thai_tokenizer
}

def translate_sentence(model, sentence):
    model.eval()  # Set model to evaluation mode
    
    # Tokenize and numericalize the input sentence
    tokenized_sentence = token_transform['en'](sentence)  # Tokenize
    src_tokens = vocab_transform['en'].lookup_indices(tokenized_sentence)  # Convert to indices
    src_tensor = torch.tensor([src_tokens]).to(device)

    bos_idx = vocab_transform['th']["<bos>"]
    eos_idx = vocab_transform['th']["<eos>"]

    max_len=50

    # Start decoding loop
    generated_tokens = [bos_idx]  # Start with <bos>
    with torch.no_grad():
        for _ in range(max_len):
            trg_tensor = torch.tensor([generated_tokens]).to(device)  # Convert to tensor
            
            # Extract output from the model
            output, _ = best_model(src_tensor, trg_tensor)  # Extract only the first value

            next_token = output.argmax(2)[:, -1].item()  # Get the highest probability token

            if next_token == eos_idx:  # Stop if <eos> is predicted
                break
            generated_tokens.append(next_token)  # Append next token

    # Convert predicted token indices back to words
    mapping = vocab_transform['th'].get_itos()
    translated_sentence = " ".join([mapping[token] for token in generated_tokens[1:]])  # Skip <bos>

    # Print results
    print("Input Sentence:", sentence)
    print("Translated Sentence:", translated_sentence)

    return translated_sentence

best_attention = 'multiplicative'
best_model = select_model(best_attention)
save_path = f'models/{best_attention}_attention.pt'
best_model.load_state_dict(torch.load(save_path))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/translate', methods=['POST'])
def translate():
    # Get the source sentence from the request JSON
    input_data = request.get_json()
    sentence = input_data.get('sentence')

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        # Call the translate function
        translated_text = translate_sentence(best_model, sentence)

        # Return the translated text in the response
        return jsonify({"translation": translated_text})

    except Exception as e:
        # In case of error, return an error message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)