import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import networkx as nx

class SocialMediaDataGenerator:
    def __init__(self,num_users = 100, num_posts = 500):
        self.num_users = num_users
        self.num_posts = num_posts
        self.users = self._generate__users()
        self.topics = ['politics','technology','sports','entertainment','science','health']

    def _generate__users(self):
        users =[]
        for i in range(self.num_users):
            users.append({
                'user_id':f'user_{i:03d}',
                'username':f'user_{i}',
                'followers':random.randint(10,10000),
                'following': random.randint(10,10000),
                'join_date':datetime(2023,1,1)+timedelta(days = random.randint(0,365)),
                'verified':random.random() < 0.1
            })
        return pd.DataFrame(users)

    def generate_posts(self):
        posts = []
        sentiments = ['positive','negative','neutral']
        post_templates ={
            'positive':[
                "Just had an amazing experience with {product}! Highly recommend",
                "So excited about the new {topic} developments.Future is bright",
                "Great to see progress in {field}. Well done everyone!",
                "Loving the new features in {product}. Game changer!"
            ],
            'negative':[
                "Really disappointed with {product}.Excepted better",
                "The state {topic} is concerning. Need urgent action",
                "Frustrated with the lack of progress in {field}.",
                "{product} needs serious improvements. Not impressed"
            ],
            'neutral':[
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
            product = random.choice(['Twitter','Facebook','Instagram','TikTok','YouTube'])
            field = random.choice(['AI','ML','data science','programming'])
            template = random.choice(post_templates[sentiment])
            content = template.format(topic = topic,product = product,field = field)
            hashtags = f" #{topic} #{product}"
            content +=hashtags

            posts.append({
                'post_id' : f'post_{i:04d}',
                'user_id': user,
                'content': content,
                'sentiment': sentiment,
                'topic':topic,
                'likes':random.randint(0,10000),
                'retweets_shares':random.randint(0,5000),
                'replies':random.randint(0,1000),
                'timestamp': datetime.now() - timedelta(hours = random.randint(0,720)),
                'hashtags': f' #{topic} #{product}',

            })
            return pd.DataFrame(posts)
    def generate_network(self):
        G = nx.DiGraph()

        for _,user in self.users.iterrows():
            G.add_node(user['user_id'],followers = user['followers'],verifed = user['verified'])
        for user_id in self.users['user_id']:
            num_following = random.randint(10,50)
            following = random.sample([u for u in self.users['user_id'] if u != user_id],min(num_following, self.num_users-1))
            for follow_id in following:
                G.add_edge(user_id,follow_id)
        return G

    def save_data(self):
        posts_df = self.generate_posts()
        network = self.generate_network()

        self.users.to_csv('user.csv',index = False)
        posts_df.to_csv('data/post.csv',index = False)

        edges_df = pd.DataFrame(list(network.edges()),columns = ['followers','followed'])
        edges_df.to_csv('data/network.csv',index = False)
        print("Generated Data...")
        print(f"-Users: {len(self.users)}")
        print(f"posts: {len(posts_df)}")
        print(f"Network Edges: {len(edges_df)}")

        return self.users, posts_df, edges_df
if __name__ == "__main__":
    generator = SocialMediaDataGenerator(num_users = 50,num_posts = 200)
    users,posts,network = generator.save_data()



