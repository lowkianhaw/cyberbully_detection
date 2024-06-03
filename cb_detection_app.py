import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from PIL import Image

# Ensure NLTK data is downloaded
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Load model and tokenizer
model_dir = 'teoh0821/cb_detection'
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

# Set up Streamlit interface
st.title("Welcome to CyberSafe - Your Guardian Against Cyberbullying!")
st.markdown("""
    If you wonder what makes a comment hurtful, check the comment here.
""")

home_image = Image.open('Picture1.jpg')
st.image(home_image, caption="")

# User input
user_input = st.text_area("Enter Text to Analyze")

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove HTML tags
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special symbols and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    
    # Remove punctuations
    punctuations = '#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p, '')
    
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join the words back into a sentence
    processed_text = ' '.join(filtered_words)
    
    return processed_text

def get_aspect(text):
    tagged = pos_tag(word_tokenize(text))
    aspects = []
    for i in range(len(tagged)-1):
        if tagged[i][1] == 'NN' and tagged[i+1][1] == 'JJ':
            aspects += [tagged[i][0], tagged[i+1][0]]
    return ' '.join(aspects)

if st.button("Analyze"):
    with st.spinner("Please wait for a few seconds, the application will be loaded soon ⏳"):
        # Preprocess the text
        processed_text = preprocess_text(user_input)
        
        # Tokenize and predict
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        
        # Map prediction to sentiment
        sentiments = ['Non-cyberbullying Post', 'Cyberbullying Post']
        sentiment = sentiments[prediction]
        
        # Print custom messages based on the sentiment
        if sentiment == "Non-cyberbullying Post":
            st.success("This message is all clear! 😊 Keep spreading positivity!")
        elif sentiment == "Cyberbullying Post":
            st.error("This message contains cyberbullying content. Let's spread kindness instead. 🚫")

