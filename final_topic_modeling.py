import pandas as pd
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from gensim.models.phrases import Phrases, Phraser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re
import datetime
import numpy as np
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Removing stopwords, punctuation, and non-alphabetic characters
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation and token.isalpha()]
    
    # Exclude specific words
    excluded_words = ['control', 'raise', 'supply', 'demand', 'staff', 'sometimes', 'ask', 'benefit', 'life', 'corporate', 'block', 'phone', 'culture', 'idea', 'help', 'next', 'different', 'situation', 'others', 'announce', 'car', 'manufacturing', 'today', 'stop', 'happening', 'made', 'enough', 'firing', 'state', 'country', 'coming', 'pretty', 'cut', 'start', 'role', 'notice', 'law', 'isnt', 'point', 'stupid', 'acquisition', 'feedback', 'energy', 'internal', 'downloader', 'hybrid', 'offer', 'looking', 'pay', 'interview', 'salary', 'freeze', 'first', 'never', 'started', 'happened', 'working', 'told', 'back', 'probably', 'post', 'put', 'wrong', 'recruiter', 'based', 'obviously', 'individual', 'service', 'please', 'based', 'cause', 'list', 'performer', 'end', 'bit', 'back', 'home', 'point', 'making', 'maybe', 'office', 'meeting', 'place', 'happen', 'open', 'mass', 'low', 'likely', 'team', 'performance', 'havent', 'yet', 'previous', 'reddit', 'slow', 'seen', 'tomorrow', 'innovation', 'mba', 'fulltime', 'market', 'rid', 'shit', 'leave', 'world', 'bad', 'fuck', 'literally', 'nothing', 'workforce', 'user', 'data', 'focus', 'crypto', 'bullish', 'retail', 'white', 'business', 'laying', 'red', 'theyre', 'flag', 'productivity', 'weve', 'season', 'announced', 'part', 'attrition', 'consulting', 'headcount', 'reason', 'right', 'high', 'wont', 'mean', 'worker', 'real', 'fed', 'rate', 'faster', 'cry', 'look', 'ten', 'send', 'lesson', 'parent', 'logical', 'announces', 'web', 'u', 'ceo', 'statement', 'letter', 'instead', 'global', 'posted', 'talent', 'ton', 'due', 'look', 'considering', 'industry', 'hour', 'shit', 'really', 'email', 'year', 'huge', 'hit', 'wouldnt', 'thats', 'billion', 'sure', 'whos', 'badly', 'change', 'tough', 'supposed', 'must', 'want', 'thing', 'layoff', 'paycut', 'open', 'employee', 'month', 'ok', 'actually', 'long', 'definitely', 'mind', 'economy', 'pretty', 'way', 'start', 'everyone', 'poor', 'revenue', 'talk', 'wont', 'company', 'stock', 'since', 'one', 'news', 'layoffs', 'job', 'get', 'say', 'like', 'think', 'work', 'make', 'time', 'employee', 'year', 'company', 'day']
    tokens = [token for token in tokens if token not in excluded_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # POS tagging and selecting nouns and adjectives
    tagged_tokens = pos_tag(tokens)
    tokens = [token for token, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('JJ')]
    
    # Joining tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def preprocessing_2(text):
    text = re.sub(r'\n\s*\n', ' ', text)  # Remove empty lines
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    return text

def convert_unix_timestamp(unix_timestamp):
    timestamp = datetime.datetime.fromtimestamp(unix_timestamp)
    formatted_date = timestamp.strftime('%Y-%m-%d')
    return formatted_date
def convert_unix_timestamp_only_year(unix_timestamp):
    timestamp = datetime.datetime.fromtimestamp(unix_timestamp)
    formatted_date = timestamp.strftime('%Y')
    return formatted_date

# Load the preprocessed dataset
dataset = pd.read_excel('preprocessed_data.xlsx')

dataset['conv_date_year'] = dataset['date'].apply(convert_unix_timestamp_only_year)

dataset_2019 = dataset[dataset['conv_date_year'] == '2019']
dataset_2020 = dataset[dataset['conv_date_year'] == '2020']
dataset_2021 = dataset[dataset['conv_date_year'] == '2021']
dataset_2022 = dataset[dataset['conv_date_year'] == '2022']
dataset_2023 = dataset[dataset['conv_date_year'] == '2023']
# negative_dataset = dataset[dataset['Sentiment'] == 'Negative']
# positive_dataset = dataset[dataset['Sentiment'] == 'Positive']
# neutral_dataset = dataset[dataset['Sentiment'] == 'Neutral']

# dataset_list = [positive_dataset, negative_dataset, neutral_dataset]

dataset_list = [dataset_2019, dataset_2020, dataset_2021, dataset_2022, dataset_2023]
for year,dataset in  zip(range(2019, 2024), dataset_list):


    dataset['conv_date'] = dataset['date'].apply(convert_unix_timestamp)

    # Replace NaN values with an empty string
    dataset['text'] = dataset['text'].fillna('')

    # Apply text preprocessing to the 'preprocessed_text' column
    dataset['preprocessed_text'] = dataset['preprocessed_text'].apply(preprocess_text)

    # Remove the specified words from the preprocessed text
    excluded_words = ['program','court','hire','loan','care','package', 'massive', 'organization','problem','hr','anything','executive','quit','question' ,'anything', 'hard', 'period','sign','group','issue', 'guy', 'project', 'age','value', 'customer', 'old', 'werent','supervisor','product','value','money','price','arent','district', 'contract', 'department', 'process','defense', 'experience', 'arent','older', 'severance', 'employer', 'great','least', 'career', 'older', 'lay','term','control', 'raise', 'supply', 'demand', 'staff','sometimes', 'ask', 'benefit', 'life','corporate', 'block', 'phone', 'culture','idea', 'help', 'next', 'different', 'situation','others','announce','car', 'manufacturing', 'today','stop', 'happening', 'made', 'enough','firing', 'state','country','coming', 'pretty','cut', 'start', 'role', 'notice', 'law','isnt','point', 'stupid', 'acquisition', 'feedback', 'energy', 'internal', 'downloader', 'hybrid', 'offer', 'looking', 'pay', 'interview', 'salary', 'freeze','first', 'never', 'started', 'happened', 'working','told', 'back', 'probably','post','put', 'wrong', 'recruiter', 'based', 'obviously', 'individual', 'service', 'please', 'based', 'cause', 'list', 'performer','end', 'bit','back', 'home', 'point', 'making','maybe', 'office', 'meeting', 'place', 'happen', 'open', 'mass','low','likely','team', 'performance','havent', 'yet', 'previous', 'reddit', 'slow', 'seen', 'tomorrow', 'innovation', 'mba', 'fulltime','market','rid', 'shit', 'leave', 'world', 'bad', 'fuck', 'literally', 'nothing','workforce','user', 'data', 'focus', 'crypto', 'bullish','retail', 'white', 'business', 'laying', 'red', 'theyre', 'flag', 'productivity', 'weve', 'season', 'announced', 'part', 'attrition', 'consulting', 'headcount','reason', 'right','high','wont','mean','worker','real','fed','rate','faster','cry','look','ten','send','lesson','parent','logical','announces','web','u','ceo','statement','letter','instead','global','posted','talent','ton','due','look','considering','industry','survive','better','visa','talk','candidate','government','come','right','board','protection','claim','required','email','medical','small','comment','weight','website','used','bot','coverage','wait','smart','dead','merger','major','look','see', 'getting', 'id', 'right','large', 'decision', 'well','need','sure','person','contractor', 'plan', 'already', 'best', 'fund', 'emergency', 'worry', 'social','blue', 'treat', 'deliver', 'awesome', 'handled', 'informed','announcing', 'reported', 'recall', 'cook','trigger','collar','younger','wave','excuse','morning','pulling','avoid','worked','x','le','avoid','youd','headline','pulling','v','read','agreed','wonder','able','sick','volunteer','fucked','disney','clause','added','file','regarding','voluntary','mean','theyre','something','position','everyone','getting','level','free','store','minute','food','sound','file','call','need','case','feel','tell','wont','current','manager','management','someone','usually','actually','bro','say','tc','always','doesnt','video', 'link','say','keep','saying','lmao','bust','pattern','op','expecting','thats','seems','article','lol','yea','quiet','info','called','good','round', 'want', 'find', 'let', 'said', 'much', 'cant', 'number', 'take', 'way', 'union', 'employee','thing', 'even', 'ive', 'really', 'going','make','also' ,'job', 'new', 'go', 'know','layoff', 'company', 'get', 'tech', 'like', 'year', 'dont', 'time', 'one', 'give', 'month', 'youre','got', 'week', 'im', 'laid', 'day', 'people', 'last', 'work', 'didnt', 'cost', 'give', 'still', 'think', 'many', 'big', 'lot']
    
    dataset['preprocessed_text'] = dataset['preprocessed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in excluded_words]))

    # Convert the preprocessed text to a list of token lists
    tokenized_text = [word_tokenize(text) for text in dataset['preprocessed_text']]

    # Create a dictionary from the tokenized text
    dictionary = corpora.Dictionary(tokenized_text)

    # Create a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in tokenized_text]
    # Define the number of topics
    num_topics = 1

    # Train the LDA model
    lda_model = gensim.models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=10)

#  # Print the top 5 topics and their top words
#     for topic_id in range(num_topics):
#         print(f"Topic {topic_id + 1} for {dataset['conv_date_year'].iloc[0]}:")
#         topic_words = lda_model.show_topic(topic_id, topn=4)
#         topic_words = [word for word, _ in topic_words]
#         print(topic_words)
#         print()


    # Get the top words for each topic
    top_words_per_topic = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=6)
        top_words = [word for word, _ in topic_words]
        top_words_per_topic.append(top_words)

    # Create word clouds for each topic
    plt.figure(figsize=(12, 8))
    for topic_id, top_words in enumerate(top_words_per_topic):
        wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(top_words))
        plt.subplot(1, num_topics, topic_id + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Topic {topic_id + 1} for Year {year}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()