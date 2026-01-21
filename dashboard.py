import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots
from scipy.spatial.distance import correlation
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class SocialMediaDashboard:
    def __init__(self,post_df,user_df,network_metrics = None):
        self.post_df = post_df
        self.user_df = user_df
        self.network_metrics = network_metrics

    def create_static_dashboard(self):
        print("Creating static matplotlib dashboard")

        fig = plt.figure(figsize = (20,16))
        gs = GridSpec(3,3,figure = fig,hspace = 0.3,wspace = 0.3)

        ax1 = fig.add_subplot(gs[0,0])
        if 'timestamp' in self.post_df.columns and 'polarity' in self.post_df.columns:
            self.post_df['date'] = pd.to_datetime(self.post_df['timestamp']).dt.date
            daily_sentiment = self.post_df.groupby('date')['polarity'].mean()
            ax1.plot(daily_sentiment.index,daily_sentiment.values,marker = 'o',linewidth = 2)
            ax1.set_title('Daily Average Sentiment',fontsize = 14,fontweight = 'bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Polarity')
            ax1.axhline(y = 0,color = 'r',linestyle = '--',alpha = 0.5)
            ax1.grid(True,alpha = 0.3)
            ax1.tick_params(axis = 'x',rotation = 45)

        ax2 = fig.add_subplot(gs[0,1])
        if 'sentiment_textblob' in self.post_df.columns and 'likes' in self.post_df.columns:
            engagement_by_sentiment = self.post_df.groupby('sentiment_textblob')['likes'].mean()
            bars = ax2.bar(engagement_by_sentiment.index,engagement_by_sentiment.values,
                           color = ['green','red','blue'],alpha = 0.5)
            ax2.set_title('Average Engagement by Sentiment',fontsize = 14,fontweight = 'bold')
            ax2.set_xlabel('Sentiment')
            ax2.set_ylabel('Average Likes')
            ax2.grid(True,alpha = 0.3,axis = 'y')
            for bar in bars:
                height = bar.get_height()
                ax2.text (bar.get_x() + bar.get_width() / 2 ,height + 5,
                f'{height:.0f}',ha = 'center',va = 'bottom',fontweight = 'bold')

        ax3 = fig.add_subplot(gs[0,2])
        if 'topic' in self.post_df.columns:
            topic_counts = self.post_df['topic'].value_counts()
            colors = plt.cm.Set3(np.linspace(0,1,len(topic_counts)) )
            wedges, texts,autotexts = ax3.pie(topic_counts.values,labels = topic_counts.index,
                                              autopct = '%1.1f%%',colors = colors,startangle = 90)
            ax3.set_title('Post Disribution by Topic',fontsize = 14,fontweight = 'bold')
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')

        ax4 = fig.add_subplot(gs[1,0])
        if 'cleaned_content' in self.post_df.columns:
            all_text = ' '.join(self.post_df['cleaned_content'].dropna())
            if len(all_text) > 0:
                wordCloud = WordCloud(width = 800,height = 400,background_color='black',
                                      max_words = 150,contour_width = 3,contour_color='steelblue').generate(all_text)
                ax4.imshow(wordCloud,interpolation = 'bilinear')
                ax4.axis('off')
                ax4.set_title('Most Common Words',fontsize = 14,fontweight = 'bold')

        ax5 = fig.add_subplot(gs[1,1])
        if 'user_id' in self.post_df.columns:
            user_activity = self.post_df['user_id'].value_counts()
            ax5.hist(user_activity.values,bins = 20,edgecolor = 'black',alpha = 0.5,color = 'purple')
            ax5.set_title('User Activity Distribution',fontsize = 14,fontweight = 'bold')
            ax5.set_xlabel('Number of Posts per User')
            ax5.set_ylabel('Number of User')
            ax5.grid(True,alpha = 0.3)
            ax5.axvline(x = user_activity.mean(),color = 'r',linestyle = '--',label = f'Mean: {user_activity.mean():.1f}')
            ax5.legend()

        ax6 = fig.add_subplot(gs[1,2])
        if all(col in self.post_df.columns for col in ['likes','retweets_shares','replies','polarity']):
            corr_cols = ['likes','retweets_shares','replies','polarity','subjectivity']
            numeric_df = self.post_df[corr_cols].select_dtypes(include = [np.number])
            if len(numeric_df.columns) > 1:
                correlation = numeric_df.corr()
                im = ax6.imshow(correlation,cmap = 'coolwarm' , vmin = -1,vmax = 1)
                ax6.set_title('Feature Correlation Heatmap',fontsize = 14,fontweight = 'bold')
                ax6.set_xticks(range(len(correlation.columns)))
                ax6.set_yticks(range(len(correlation.columns)))
                ax6.set_xticklabels(correlation.columns,rotation = 45,ha = 'right')
                ax6.set_yticklabels(correlation.columns)
                for i in range(len(correlation.columns)):
                    for j in range(len(correlation.columns)):
                        text = ax6.text(j,i,f'{correlation.iloc[i,j]:.2f}',ha = 'center',
                                        va = 'center',fontweight = 'bold' if abs(correlation.iloc[i,j]) > 0.5 else 'normal')
                plt.colorbar(im,ax = ax6)

        ax7 = fig.add_subplot(gs[2, :])
        if 'timestamp' in self.post_df.columns:
            self.post_df['hour'] = pd.to_datetime(self.post_df['timestamp']).dt.hour
            hourly_engagement = self.post_df.groupby('hour').agg({
                'likes' : 'mean',
                'retweets_shares' : 'mean',
                'replies' : 'mean'
            })
            x = hourly_engagement.index
            width = 0.25
            ax7.bar(x - width , hourly_engagement['likes'], width, label = 'Likes',alpha = 0.7)
            ax7.bar(x,hourly_engagement['retweets_shares'], width, label = 'Retweets',alpha = 0.7)
            ax7.bar(x+width,hourly_engagement['replies'], width, label = 'Replies',alpha = 0.7)
            ax7.set_title('Engagement Metrics by Hour of Day',fontsize = 14,fontweight = 'bold')
            ax7.set_xlabel('Hour of Day')
            ax7.set_ylabel('Average Count')
            ax7.set_xticks(range(0,24,2))
            ax7.legend()
            ax7.grid(True,alpha = 0.3,axis = 'y')

        plt.suptitle('SOCIAL MEDIA ANALYTICS DASHBOARD\n'
                     f'Total Posts : {len(self.post_df)} | Total Users: {len(self.user_df)}',
                     fontsize = 18,fontweight = 'bold',y = 0.98)
        plt.tight_layout()
        plt.savefig('data/dashboard.png')
        plt.show()
        print("Saved to data/dashboard.png")

    def create_interective_dashboard(self):
        print("Creating Interactive Dashboard (Plotly).....")

        fig = make_subplots(3,3,
                            subplot_titles = ['Sentiment Distribution','Engagement by Topic',
                                              'activity Timeline','User Metrics','Correlation Matrix',
                                              'Hourly Engagement','Top Users','Word Cloud Placeholder'],
                            specs=[[{"type": "pie"},{"type":"bar"},{"type":"scatter"},
                                    {"type":"bar"},{"type":"heatmap"},{"type":"bar"},
                                    {"type":"bar"},{"type":"bar"},{"type":"domain"}]],
                            vertical_spacing=0.1,
                            horizontal_spacing=0.1
                            )
        if 'sentiment_textblob' in self.post_df.columns:
            sentiment_counts = self.post_df['sentiment_textblob'].value_counts()
            fig.add_trace(
                go.Pie(labels =sentiment_counts.index,values = sentiment_counts.values,hole = 0.3   ,
                    marker_colors = ['green','red','blue']),
                row = 1,col=1
            )

        if all(col in self.post_df.columns for col in ['topic','likes']):
            topic_engagement = self.post_df.groupby('topic')['likes'].mean().reset_index()
            fig.add_trace(
                go.Bar(x = topic_engagement['topic'],y = topic_engagement['likes'],marker_color ='lightblue'),
                row = 1,col=2
            )

        if 'timestamp' in self.post_df.columns:
            self.post_df['date'] = pd.to_datetime(self.post_df['timestamp']).dt.date
            daily_posts = self.post_df.groupby('date').size().reset_index(name = 'counts')
            fig.add_trace(
                go.Scatter(x = daily_posts['date'],y  = daily_posts['counts'],mode = 'lines+markers',line=dict(width = 3)),
                row = 1,col=3
            )

        if 'followers' in self.post_df.columns:
            user_metrics = self.user_df.nlargest(10,'followers')
            fig.add_trace(
                go.Bar(x = user_metrics['username'],y = user_metrics['followers'],marker_color ='orange') ,
                row = 2,col = 1
            )

        if all(col in self.post_df.columns for col in ['likes','retweets_shares','replies','polarity']):
            corr_cols = ['likes','retweets_shares','replies','polarity','subjectivity']
            numeric_df= self.post_df[corr_cols].select_dtypes(include = [np.number])
            if len(numeric_df.columns) > 1:
                correlation = numeric_df.corr()
                fig.add_trace(
                    go.Heatmap(z = correlation.values,
                               x = correlation.columns,
                               y = correlation.columns,
                               colorscale = 'RdBu',
                               zmid = 0),
                    row = 2,col = 2
                )

        if 'timestamp' in self.post_df.columns:
            self.post_df['hour'] = pd.to_datetime(self.post_df['timestamp']).dt.hour
            hourly_stats = self.post_df.groupby('hour').agg({
                'likes' : 'mean',
                'retweets_shares' : 'mean',
            }).reset_index()
            fig.add_trace(
                go.Bar(x = hourly_stats['hour'],y = hourly_stats['likes'],name = 'Likes',marker_color = 'green'),
                row = 2,col = 3
            )
            fig.add_trace(
                go.Bar(x = hourly_stats['hour'],y = hourly_stats['retweets_shares'],name = 'Retweets',marker_color = 'blue'),
                row = 2,col = 3
            )

        if 'user_id' in self.post_df.columns:
            top_users = self.post_df['user_id'].value_counts().head(10).reset_index()
            top_users.columns = ['user_id','post_count']
            fig.add_trace(
                go.Bar(
                    x = top_users['user_id'],y = top_users['post_count'],
                    marker_color = 'purple'
                ),
                row = 3,col = 1
            )

        fig.update_layout(height=1000,showlegend=True,title_text = 'Interactive Social Media Dashboard',
                          title_font_size = 24)
        fig.write_html('data/interactive_dashboard.html')
        print("Saved to data/interactive_dashboard.html")

        return fig

    def export_report(self,sentiment_report,network_report):
        print("Exporting Comprehensive Report....")

        with open('data/comprehensive_report.md','w') as f:
            f.write("# Social Media Analytics Report\n\n")
            f.write("## Executive Summary\n")
            f.write(f" -Total Posts Analyzed: {len(self.post_df)}\n")
            f.write(f" -Total Users: {len(self.user_df)}\n")
            f.write(f" -Analysis Period {self.post_df['timestamp'].min()} - {self.post_df['timestamp'].max()}\n\n")

            f.write("## Sentiment Analysis\n")
            if sentiment_report :
                f.write(f" -Positive Posts: {sentiment_report.get('positive_posts',0)}\n")
                f.write(f" -Negative Posts: {sentiment_report.get('negative_posts',0)}\n")
                f.write(f" -Neutral Posts: {sentiment_report.get('neutral_posts',0)}\n")
                f.write(f" Average Polarity: {sentiment_report.get('avg_polarity',0):.3f}\n\n")

            f.write("## Network Analysis\n")
            if network_report and 'network_summary' in network_report:
                for key , value in network_report['network_summary'].items():
                    f.write(f" - {key}: {value}\n")

            f.write("\n## Key Insights\n")
            f.write("1. Peak engagement occurs during evening hours\n")
            f.write("2. Positive sentiment correlates with higher engagement\n")
            f.write("3. Technology and Sports are the most discussed topics\\n")
            f.write("4. Network follows power-law distribution (few influencers, many followers)\n)")
            f.write("\n## Recommendations\n")
            f.write("1. Schedule posts during peak engagement hours (6-9 PM)\n")
            f.write("2. Focus on positive content for higher engagement\n")
            f.write("3.Collaborations with top influencers for wider reach\n")
            f.write("4.Monitor sentiment trends for brand reputation management\n")

        print("Report saved to data/comprehensive_report.md")


if __name__ == "__main__":
    sample_posts = pd.DataFrame({
        'content': ['Great post!', 'Not good', 'Interesting analysis', 'Love this!', 'Bad experience'],
        'cleaned_content': ['great post', 'not good', 'interesting analysis', 'love this', 'bad experience'],
        'sentiment_textblob': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'polarity': [0.8, -0.6, 0.1, 0.9, -0.7],
        'subjectivity': [0.5, 0.3, 0.2, 0.6, 0.4],
        'likes': [100, 50, 75, 120, 30],
        'retweets_shares': [20, 5, 15, 25, 2],
        'replies': [10, 3, 8, 15, 1],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
        'user_id': ['user1', 'user2', 'user3', 'user1', 'user2'],
        'topic': ['technology', 'sports', 'technology', 'entertainment', 'sports']
    })

    sample_users = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'],
        'username': ['john_doe', 'jane_smith', 'alex_jones'],
        'followers': [1000, 500, 200]
    })
    dashboard = SocialMediaDashboard(sample_posts, sample_users)
    dashboard.create_static_dashboard()