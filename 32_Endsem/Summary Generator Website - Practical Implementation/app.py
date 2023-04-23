from flask import Flask, render_template, request
import re
import string
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)
app.use_static_for_template = True
def generate_summary_t5(text, max_length=100):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Tokenize the article text
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=max_length,  # Use max_length parameter here
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary



# Define a function to correct the spellings in a list of summaries
def correct_summaries(summaries):
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('Downloads/en_model.bin')
    corrected_summaries = []
    for summary in summaries:
        corrected_summary = corrector.FixFragment(summary)
        corrected_summaries.append(corrected_summary)
    return corrected_summaries



def preprocess_text(text):
    # Remove HTML tags and URLs
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Tokenize the text using NLTK
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Lemmatize the tokens using spaCy
    lemmas = [token.lemma_ for token in nlp(" ".join(tokens))]

    # Remove any remaining non-alphabetic tokens
    lemmas = [lemma for lemma in lemmas if lemma.isalpha()]

    # Join the lemmas back into a string
    text = " ".join(lemmas)

    return text
@app.route('/')
def index():
    return render_template('index.html', summary="")

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    max_length = int(request.form['max_length'])
    summary = generate_summary_t5(text, max_length=max_length)  # Pass max_length as argument
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)