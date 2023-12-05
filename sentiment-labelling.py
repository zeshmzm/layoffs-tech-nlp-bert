from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import re
import datetime
import numpy as np

df = pd.read_csv(r"layoff_comments_list.csv", header=0, encoding='latin-1')

def preprocessing_1(data):


    # Create an empty list to store the processed text values
    processed_texts = []

    # Iterate over the 'text' column and process each comment
    for reddit_comment in data['text']:
        comment_words = []
        
        for word in reddit_comment.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = "user"
            elif 'Â' in word:
                word = word.replace('Â', '')
            elif word.startswith("http"):
                word = 'http'
            elif word.startswith('>') or word.startswith('<') or word.startswith('.'):
                word = ''
            comment_words.append(word)

        while ('' in comment_words):
            comment_words.remove('')
        while ('http' in comment_words):
            comment_words.remove('http')

        processed_text = ' '.join(comment_words)
        processed_texts.append(processed_text)

    # Replace the 'text' column with the processed text values
    data['text'] = processed_texts

    return data

def preprocessing_2(text):
    text = re.sub(r'\n\s*\n', ' ', text)  # Remove empty lines
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    return text

def convert_unix_timestamp(unix_timestamp):
    timestamp = datetime.datetime.fromtimestamp(unix_timestamp)
    formatted_date = timestamp.strftime('%Y-%m-%d')
    return formatted_date

df = preprocessing_1(df)


df['text'] = df['text'].apply(preprocessing_2)
df_text = df['text']

df['conv_date'] = df['date'].apply(convert_unix_timestamp)

df_date = df['conv_date']
# for word in reddit_comment.split(' '):
#     if word.startswith('@') and len(word) > 1:
#         word = "user"
#     elif word.startswith("http"):
#         word = 'http'
#     elif word.startswith('>') or word.startswith('<') or word.startswith('.'):
#         word = ''
#     comment_words.append(word)

# while ('' in comment_words):
#     comment_words.remove('')
# while ('http' in comment_words):
#     comment_words.remove('http')

# reddit_proc = ' '.join(comment_words)

# Load model and tokenizer
bertweet = "finiteautomata/bertweet-base-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(bertweet)
tokenizer = AutoTokenizer.from_pretrained(bertweet)
labels = ['NEG', 'NEU', 'POS']



# Assign sentiment labels to each text in df_text
sentiments = []

for i, text in enumerate(df_text):
    try:
                # Load model and tokenizer
        bertweet = "finiteautomata/bertweet-base-sentiment-analysis"
        model = AutoModelForSequenceClassification.from_pretrained(bertweet)
        tokenizer = AutoTokenizer.from_pretrained(bertweet)
        labels = ['NEG', 'NEU', 'POS']

        # Encode and truncate the text while applying padding
        encoded_comment = tokenizer.encode_plus(text, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
        input_ids = encoded_comment['input_ids']
        attention_mask = encoded_comment['attention_mask']

        # Pass the input tensors to the model
        output = model(input_ids, attention_mask)

        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        print(i)
        max_index = np.argmax(scores)

        if max_index == 0:
            sentiment = 'Negative'
        elif max_index == 1:
            sentiment = 'Neutral'
        elif max_index == 2:
            sentiment = 'Positive'
        else:
            sentiment = 'Undefined'
        sentiments.append(sentiment)

    except Exception as e:
        print(f"Error in Row {i+1}: {e}")

# Add the Sentiment column to the DataFrame
df['Sentiment'] = sentiments

# Print the DataFrame with the Sentiment column
df.to_csv('labelled_comments_testing123.csv', index=False)
print("Labelling Done and File Exported")

