import os
import pandas as pd
from nltk import accuracy

from data_collector import SocialMediaDataGenerator
from text_processor import TestProcessor
from sentiment_analyzer import SentimentAnalyzer
from network_analyzer import NetworkAnalyzer
from dashboard import SocialMediaDashboard

def setup_project():
    directors = ['data','output','logs']
    for directory in directors:
        os.makedirs(directory,exist_ok=True)
    print("Project directories created")

def main():
    print("Social media analytics project".upper())
    print("--"*30)

    setup_project()

    print("data generation".upper())
    generator = SocialMediaDataGenerator(num_users = 200,num_posts = 1000)
    user_df,post_df,network = generator.save_data()

    print("\n text processing".upper())
    processor = TestProcessor()
    processed_df = processor.process_dataframe(post_df,'content')

    print("sentiment analysis".upper())
    analyzer = SentimentAnalyzer()
    df_with_sentiment,sentiment_report,sentiment_by_topic = analyzer.create_sentiment_report(processed_df)

    try:
        accuracy = analyzer.train_ml_model(df_with_sentiment,'cleaned_content','sentiment')
        print(f"ML Model Accuracy: {accuracy:.2%}")
    except Exception as error:
        print(f"Note: ML training skipped - {error}")

    analyzer.visualize_sentiment(df_with_sentiment)

    print("\nnetwork analysis".upper())
    network_analyzer = NetworkAnalyzer()
    network_analyzer.load_network()
    network_report = network_analyzer.generate_network_report()
    network_analyzer.visualization_network(max_nodes=100)

    print("\nDashboard creation".upper())
    dashboard = SocialMediaDashboard(df_with_sentiment,user_df,network_report)
    dashboard.create_static_dashboard()

    try:
        dashboard.create_static_dashboard()
    except Exception as error:
        print(f"Note: Interactive Dashboard creation skipped - {error}")

    dashboard.export_report(sentiment_report,network_report)

    print("\n" + "-"*60)
    print("Project completed".upper())
    print("-"*60)
    print(f"Data: {len(user_df)} users, {len(post_df)} posts")
    print(f"Analysis: Sentiment, Network, Engagement Metrics")
    print("Outputs saved to 'data/'directory': '")
    print("- sentiment_analysis.png")
    print("-network_visualization.png")
    print("-degree_distribution.png")
    print("-dashboard.png")
    print(" -interactive_dashboard.png")
    print(" -comprehensive_report.md")

    return df_with_sentiment,user_df,network_report

if __name__ == "__main__":
    df,users , report = main()