import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from textblob.en import subjectivity, sentiment

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000,stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100,random_state=42)

    def analyze_with_textblob(self,text):
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment_textblob':'positive' if blob.sentiment.polarity > 0.1 else 'negative'
            if blob.sentiment.polarity <-0.1 else 'neutral'
        }

    def analyze_batch(self,texts):
        results = []
        for text in texts:
            if isinstance(text,str):
                result = self.analyze_with_textblob(text)
                results.append(result)
            else:
                results.append({'polarity': 0,'subjectivity': 0,'sentiment_textblob':"neutral"})
        return pd.DataFrame(results)

    def train_ml_model(self,df,text_column = 'cleaned_content',label_column = 'sentiment'):
        print("Training ML sentiment model.....")
        X = self.vectorizer.fit_transform(df[text_column])
        y = df[label_column]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

        self.classifier.fit(X_train,y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)

        print(f"Model trained with accuracy: {accuracy:.2%}")
        print("\nClassification report:")
        print(classification_report(y_test,y_pred))
        return accuracy

    def predict_sentiment(self,texts):
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Model not trained yet. Call train_ml_model first.")
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        return predictions,probabilities

    def create_sentiment_report(self,df):
        print("Generating sentiment report...")

        sentiment_results = self.analyze_batch(df['content'])
        df = pd.concat([df,sentiment_results],axis=1)

        report = {
            'total_posts': len(df),
            'positive_posts':len(df[df['sentiment_textblob']=='positive']),
            'negative_posts':len(df[df['sentiment_textblob']=='negative']),
            'neutral_posts':len(df[df['sentiment_textblob']=='neutral']),
            'avg_polarity':df['polarity'].mean(),
            'avg_subjectivity':df['subjectivity'].mean(),
            'most_positive_post':df.loc[df['polarity'].idxmax(),'content'][:100] if len(df) >0 else ' ',
            'most_negative_post':df.loc[df['polarity'].idxmin(),'content'][:100] if len(df) >0 else ' '
        }

        sentiment_by_topic = df.groupby('topic')['sentiment_textblob'].value_counts().unstack()

        print("\nSENTIMENT REPORT: ")
        print(f"Total posts: {report['total_posts']}")
        print(f"Positive Posts: {report['positive_posts']} ({report['positive_posts']/report['total_posts']*100:.1f}%)")
        print(f"Negative posts: {report['negative_posts']} ({report['negative_posts']/report['total_posts']*100:.1f}%)")
        print(f"Neutral Posts: {report['neutral_posts']} ({report['neutral_posts']/report['total_posts']*100:.1f}%))")
        print(f"Average polarity: {report['avg_polarity']}")
        print(f"Average subjectivity: {report['avg_subjectivity']}")

        return df ,report,sentiment_by_topic

    def visualize_sentiment(self,df):
        fig,axes = plt.subplots(2,2,figsize=(12,10))
        sentiment_counts = df['sentiment_textblob'].value_counts()
        axes[0,0].pie(sentiment_counts.values,labels = sentiment_counts.index,autopct = '%1.1f%%')
        axes[0,0].set_title('Sentimental Analysis')

        axes[0,1].hist(df['polarity'],bins = 'auto' , edgecolor = 'black',alpha = 0.7)
        axes[0,1].set_xlabel('Polarity')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Polarity Distribution')
        axes[0,1].axvline(x = 0, color = 'r', linestyle = '--',alpha = 0.5)

        sentiment_by_topic = df.groupby('topic')['sentiment_textblob'].value_counts().unstack()
        sentiment_by_topic.plot(kind = 'bar',ax = axes[1,0])
        axes[1,0].set_title('Sentimental by Topic')
        axes[1,0].set_xlabel('Topic')
        axes[1,0].set_ylabel('Count')
        axes[1,0].legend(title = 'Sentiment')
        axes[1,0].tick_params(axis = 'x',rotation = 45)

        scatter = axes[1,1].scatter(df['polarity'],df['likes'],c=df['subjectivity'],alpha = 0.6,cmap = 'plasma')
        axes[1,1].set_xlabel('Polarity')
        axes[1,1].set_ylabel('Likes')
        axes[1,1].set_title('Engagement Vs Sentiment')
        axes[1,1].axvline(x = 0, color = 'r', linestyle = '--',alpha = 0.5)
        plt.colorbar(scatter,ax = axes[1,1],label = 'Subjectivity')
        plt.tight_layout()
        plt.savefig('data/sentiment.png',dpi = 600,bbox_inches = 'tight')
        plt.show()
        print('Saved to data/sentiment.png')

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    test_texts = [
        "I love this product! It's amazing",
        "This is terrible. Worst experience ever",
        "The data shows interesting trends"
    ]
    results = analyzer.analyze_batch(test_texts)
    print(results)
    df_for_viz = pd.DataFrame({
        'content': test_texts,
        'topic': ['product', 'experience', 'data'],  # Required column
        'likes': [100, 50, 75],  # Required column for scatter plot
        'sentiment_textblob': results['sentiment_textblob'],
        'polarity': results['polarity'],
        'subjectivity': results['subjectivity']
    })

    # Now you can visualize
    analyzer.visualize_sentiment(df_for_viz)







