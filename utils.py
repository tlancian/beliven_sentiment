import re
import bs4
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def clean_review(text):
    # Replace HTML line breaks with a space
    pattern1 = "<br /><br />"
    text = re.sub(pattern1, ' ', text)
    
    # Remove non-alphanumeric characters (including standard punctuation and parentheses)
    pattern2 = r'[^a-zA-Z0-9\s.,!?;:"\'()\-]'
    text = re.sub(pattern2, ' ', text)
    
    # Replace multiple whitespaces with a single space
    pattern3 = r'\s+'
    cleaned_text = re.sub(pattern3, ' ', text)

    return cleaned_text



contractions_dict = {
    # Standard contractions
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he's": "he is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",

    # Slang contractions
    "ain't": "am not",
    "gimme": "give me",
    "gonna": "going to",
    "gotta": "got to",
    "kinda": "kind of",
    "lemme": "let me",
    "outta": "out of",
    "sorta": "sort of",
    "wanna": "want to",
    "cuz": "because",
    "dunno": "do not know",
    "gimme": "give me",
    "gotcha": "got you",
    "lemme": "let me",
    "shoulda": "should have",
    "gimme": "give me",
    "gotcha": "got you",
    "lemme": "let me",
    "shoulda": "should have",
    "wassup": "what is up",
    "wanna": "want to",
    "whatcha": "what are you",
    "y'all": "you all",
    "gotta": "got to",
    "gimme": "give me",
    "imma": "I am going to",
    "innit": "isn't it",
    "kinda": "kind of",
    "nite": "night",
    "outta": "out of",
    "prolly": "probably",
    "s'more": "some more",
    "somethin'": "something",
    "sorta": "sort of",
    "tryna": "trying to",
    "wassup": "what is up",
    "whatcha": "what are you",
    "y'all": "you all",
    "y'know": "you know"
}

def expand_contractions(text):
    # Replace contractions with their expanded forms
    for contraction, expanded_form in contractions_dict.items():
        pattern = r'\b' + re.escape(contraction) + r'\b'
        text = re.sub(pattern, expanded_form, text)
    return text

def text_cleaner(text):
    # Convert text to lowercase
    text = text.lower()

    # Expand contractions
    text = expand_contractions(text)
    
    # Remove HTML tags
    text = bs4.BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return filtered_words