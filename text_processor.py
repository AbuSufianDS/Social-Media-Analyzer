import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TestProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.emotion_words = {
            'positive':['love','great','amazing','excellent','happy','good','best'],
            'negative': ['hate','bad','terrible','awful','sad','worst','disappointed'],
            'neutral': ['article','study','data','analysis','report','research']
        }
    def clean_text(self, text):
        if not isinstance(text,str):
            return " "
        text = text.lower()

        text = re.sub(r"https?://\S+|www\.\S+"," ",text)
        text = re.sub(r"@\w+"," ",text)
        text = re.sub(r"#"," ",text)
        text = re.sub(r"[^a-zA-Z\s]"," ",text)
        text = re.sub(r"\s+"," ",text)
        return text

    def tokenize_and_lemmatize(self,text):
        tokens = word_tokenize(text)
        tokens =[self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens

    def extract_features(self,text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)

        features = {
            'text_length': len(text),
            'cleaned_length': len(cleaned),
            'word_count': len(tokens),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'has_questions': 1 if '?' in text else 0,
            'has_exclamations': 1 if '!' in text else 0,
            'has_mentions': 1 if '@' in text else 0,
            'has_hashtags': 1 if '#' in text else 0
        }
        for emotion , words in self.emotion_words.items():
            features[f'{emotion}_words'] = sum(1 for word in tokens if word in words)
        return features

    def process_dataframe(self,df,text_column = 'content'):
        print("Processing text data.....")
        df['cleaned_content'] = df[text_column].apply(self.clean_text)
        df['tokens'] = df['cleaned_content'].apply(self.tokenize_and_lemmatize)

        features_list = [ ]
        for text in df[text_column]:
            features_list.append(self.extract_features(text))
        features_df =  pd.DataFrame(features_list)
        result_df = pd.concat([df.reset_index(drop=True),features_df.reset_index(drop=True)],axis=1)
        print(f"Processed {len(df)} texts")
        print(f"Average word count: {result_df['word_count'].mean():.1f}")
        return result_df

if __name__ == "__main__":
    processor = TestProcessor()
    test_text = "Just had an amazing experience with @you mili! Highly recommend #friend #partner"
    print("Original : ", test_text)
    print("Cleaned:",processor.clean_text(test_text))
    print("Tokens : ",processor.tokenize_and_lemmatize(test_text))
    print("Features : ",processor.extract_features(test_text))





