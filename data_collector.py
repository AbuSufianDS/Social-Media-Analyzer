import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import networkx as nx
import requests


class SocialMediaDataGenerator:
    def __init__(self, num_users=100, num_posts=500):
        self.num_users = num_users
        self.num_posts = num_posts
        self.users = self._generate__users()
        self.topics = ['politics', 'technology', 'sports', 'entertainment', 'science', 'health']

    def _generate__users(self):
        users = []
        for i in range(self.num_users):
            users.append({
                'user_id': f'user_{i:03d}',
                'username': f'user_{i}',
                'followers': random.randint(10, 10000),
                'following': random.randint(10, 10000),
                'join_date': datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)),
                'verified': random.random() < 0.1
            })
        return pd.DataFrame(users)

    def generate_posts(self):
        posts = []
        sentiments = ['positive', 'negative', 'neutral']
        post_templates = {
            'positive': [
                "Just had an amazing experience with {product}! Highly recommend",
                "So excited about the new {topic} developments.Future is bright",
                "Great to see progress in {field}. Well done everyone!",
                "Loving the new features in {product}. Game changer!"
            ],
            'negative': [
                "Really disappointed with {product}.Excepted better",
                "The state {topic} is concerning. Need urgent action",
                "Frustrated with the lack of progress in {field}.",
                "{product} needs serious improvements. Not impressed"
            ],
            'neutral': [
                "Interesting article about {topic}.Worth reading.",
                "New study shows trends iin {field}. Data attached.",
                "Discussion about {product} features.What do you think?",
                "Änalysis of recent {topic} developments."
            ]
        }
        for i in range(self.num_posts):
            user = random.choice(self.users['user_id'].tolist())
            sentiment = random.choice(sentiments)
            topic = random.choice(self.topics)
            product = random.choice(['Twitter', 'Facebook', 'Instagram', 'TikTok', 'YouTube'])
            field = random.choice(['AI', 'ML', 'data science', 'programming'])
            template = random.choice(post_templates[sentiment])
            content = template.format(topic=topic, product=product, field=field)
            hashtags = f" #{topic} #{product}"
            content += hashtags

            posts.append({
                'post_id': f'post_{i:04d}',
                'user_id': user,
                'content': content,
                'sentiment': sentiment,
                'topic': topic,
                'likes': random.randint(0, 10000),
                'retweets_shares': random.randint(0, 5000),
                'replies': random.randint(0, 1000),
                'timestamp': datetime.now() - timedelta(hours=random.randint(0, 720)),
                'hashtags': f' #{topic} #{product}',

            })
        return pd.DataFrame(posts)

    def generate_network(self):
        G = nx.DiGraph()

        for _, user in self.users.iterrows():
            G.add_node(user['user_id'], followers=user['followers'], verifed=user['verified'])
        for user_id in self.users['user_id']:
            num_following = random.randint(10, 50)
            following = random.sample([u for u in self.users['user_id'] if u != user_id],
                                      min(num_following, self.num_users - 1))
            for follow_id in following:
                G.add_edge(user_id, follow_id)
        return G


    def collect_real_facebook_data(self, access_token, limit=10):
        try:
            print(f" Connecting to Facebook API with token...")

            url = "https://graph.facebook.com/v18.0/me/posts"
            params = {
                'access_token': access_token,
                'fields': 'id,message,created_time,likes.summary(true),comments.summary(true),shares',
                'limit': limit
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                posts_data = []

                if 'data' in data:
                    for i, post in enumerate(data['data']):
                        posts_data.append({
                            'post_id': post.get('id', f'fb_real_{i}'),
                            'content': post.get('message', 'No content'),
                            'likes': post.get('likes', {}).get('summary', {}).get('total_count', 0),
                            'comments': post.get('comments', {}).get('summary', {}).get('total_count', 0),
                            'shares': post.get('shares', {}).get('count', 0),
                            'timestamp': post.get('created_time', datetime.now().isoformat()),
                            'platform': 'facebook',
                            'source': 'real_api'
                        })

                    real_df = pd.DataFrame(posts_data)


                    real_df.to_csv('data/real_facebook_data.csv', index=False)
                    print(f"SUCCESS! Got {len(real_df)} REAL Facebook posts!")
                    print(" Saved to: data/real_facebook_data.csv")

                    return real_df
                else:
                    print(" No posts found in your Facebook account")
                    return None
            else:
                print(f" Facebook API Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f" Error: {e}")
            return None

    def save_data(self):
        posts_df = self.generate_posts()
        network = self.generate_network()

        self.users.to_csv('user.csv', index=False)
        posts_df.to_csv('data/post.csv', index=False)

        edges_df = pd.DataFrame(list(network.edges()), columns=['follower', 'followed'])
        edges_df.to_csv('data/network.csv', index=False)
        print("Generated Data...")
        print(f"-Users: {len(self.users)}")
        print(f"posts: {len(posts_df)}")
        print(f"Network Edges: {len(edges_df)}")

        return self.users, posts_df, edges_df


if __name__ == "__main__":
    generator = SocialMediaDataGenerator(num_users=50, num_posts=200)
    users, posts, network = generator.save_data()


    YOUR_FACEBOOK_TOKEN = "EAAWp7KggTIcBQkfHb2ZCFN2Nm4sIXU2ADJyzM9stdFZABlu5EJZCFNiw0d7glAvMkOPmDFW5ZB6erSrR7kKzzwTyhT6jEWvor836Wt0qLhSA369GGJpqt5WPCWGMNY7pR9K8QTZAybbfcA3hQuBBMvXfQEdi0BGU01cZBGQNGz3LUGgdlQSz4qcDZCcpZArs3KBKNAEbVhEvLNQdG8t79ZCH0zvCCFwNqhSUtxLmhvpIxrZBG6dxaZC7O4FCXhdDvqmP7bAvSszNf3Mn551gqOyMzmnWaW9uUx9Sjr5H1K2ZAdXHhvBERZA7r7s2RYzqvcvdOD9lFJick6GOunHBX"  # ← Paste your token here

    if YOUR_FACEBOOK_TOKEN != "EAAD...":
        print("\n" + "=" * 50)
        print(" FETCHING REAL FACEBOOK DATA...")
        print("=" * 50)

        real_data = generator.collect_real_facebook_data(
            access_token=YOUR_FACEBOOK_TOKEN,
            limit=20
        )

        if real_data is not None:
            print("\n REAL FACEBOOK DATA SAMPLE:")
            print(real_data[['post_id', 'content', 'likes', 'comments']].head())
    else:
        print("\n  Add your Facebook token to collect real data!")
        print("Get token from: https://developers.facebook.com/tools/explorer/")
