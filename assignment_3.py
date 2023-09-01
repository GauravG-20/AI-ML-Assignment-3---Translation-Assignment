from transformers import MarianTokenizer, MarianMTModel
import nltk
import spacy
from nltk import pos_tag
nltk.download('punkt')
nlp = spacy.load("en_core_web_md")

# Load the pre-trained Marian model and tokenizer for English-to-Hindi translation
translation_model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

model_name_hi_en = "Helsinki-NLP/opus-mt-hi-en"
tokenizer_hi_en = MarianTokenizer.from_pretrained(model_name_hi_en)
model_hi_en = MarianMTModel.from_pretrained(model_name_hi_en)

# Function to translate English sentence to Hindi
def translate_to_hindi(english_sentence):
    inputs = tokenizer(english_sentence, return_tensors="pt")
    translated = translation_model.generate(**inputs)
    translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_sentence

def translate_to_english(hindi_sentence):
    inputs = tokenizer_hi_en(hindi_sentence, return_tensors="pt")
    translated = model_hi_en.generate(**inputs)
    translated_sentence = tokenizer_hi_en.decode(translated[0], skip_special_tokens=True)
    return translated_sentence

def extract_and_translate_nouns(english_sentence):
    # Tokenize the English sentence
    english_words = nltk.word_tokenize(english_sentence)
    
    # Perform POS tagging to identify nouns
    pos_tags = pos_tag(english_words)
    
    noun_translations = {}
    
    for word, pos in pos_tags:
        if pos in ["NN","NNP","NNS"]:  # Check if the word is a noun
            lemmatized_word = nlp(word)[0].lemma_
            noun_translations[lemmatized_word] = word.lower()
    
    return noun_translations

# Replace specific Hindi words with their English counterparts
def replace_hindi_with_english(hindi_sentence, noun_translations):
    hindi_words = hindi_sentence.split()
    for i in range(len(hindi_words)):
        english_word = translate_to_english(hindi_words[i]).lower()
        lemmatized = nlp(english_word)[0].lemma_
        
        if lemmatized in noun_translations:
            hindi_words[i] = noun_translations[lemmatized]
    
    final_sentence = ' '.join(hindi_words)
    
    return final_sentence

# User Input
english_sentence = input("Enter english sentence: ")
hindi_translation = translate_to_hindi(english_sentence)
noun_translations = extract_and_translate_nouns(english_sentence)
final_hindi_sentence = replace_hindi_with_english(hindi_translation, noun_translations)

print("\nOriginal English sentence: ", english_sentence)
print("Final Hinglish sentence: ", final_hindi_sentence, "\n")
