import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import community as community_louvain
from collections import Counter

class NetworkAnalyzer:
    def __init__(self):
        self.G = None

    def load_network(self,edge_file = 'data/network.csv',user_file = 'user.csv'):
        edge_df = pd.read_csv(edge_file)
        user_df = pd.read_csv(user_file)
        self.G = nx.from_pandas_edgelist(edge_df,source = 'follower',
                                         target = 'followed',create_using=nx.DiGraph())

        for _,user in user_df.iterrows():
            if user['user_id'] in self.G.nodes():
                self.G.nodes[user['user_id']]['followers'] = user['followers']
                self.G.nodes[user['user_id']]['verified'] = user['verified']

        print("Network loaded")
        print(f" -Nodes: {self.G.number_of_nodes()}")
        print(f" -Edges: {self.G.number_of_edges()})")
        print(f" -Density: {nx.density(self.G):.4f}")
        return self.G

    def calculate_metrics(self):
        if self.G is None:
            raise ValueError("Network not loaded")
        metrics = {}

        metrics['num_nodes'] = self.G.number_of_nodes()
        metrics['num_edges']  = self.G.number_of_edges()
        metrics['density'] = nx.density(self.G)

        degrees = dict(self.G.degree())
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())

        metrics['avg_degree'] = np.mean(list(degrees.values()))
        metrics['max_in_degree'] = max(in_degrees.values())
        metrics['max_out_degree'] = max(out_degrees.values())

        metrics['degree_centrality'] = nx.degree_centrality(self.G)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(self.G)
        metrics['closeness_centrality'] = nx.closeness_centrality(self.G)

        Top_Influencers = sorted(in_degrees.items(),key = lambda x:x[1],reverse=True)[:10]
        metrics['Top_Influencers'] = Top_Influencers

        G_undirected = self.G.to_undirected()
        partition = community_louvain.best_partition(G_undirected)
        metrics['communities'] = partition
        metrics['num_communities'] = len(set(partition.values()))

        metrics['avg_clustering'] = nx.average_clustering(self.G)
        metrics['is_connected'] = nx.is_weakly_connected(self.G)
        metrics['avg_path_length'] = nx.average_shortest_path_length(G_undirected)  if nx.is_connected(G_undirected) else float('inf')


        return metrics

    def visualization_network(self,max_nodes = 500):
        if self.G is None:
            raise ValueError("Network not loaded")
        print("Creating network visualization")

        if len(self.G) > max_nodes:
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.items(),key = lambda  x: x[1] ,reverse=True)[:max_nodes]
            top_nodes_ids = [node for node, _ in top_nodes]
            G_viz = self.G.subgraph(top_nodes_ids).copy()
            print(f"Using subgraph {len(G_viz)} top nodes")
        else:
            G_viz = self.G

        degrees = dict(G_viz.degree())
        node_sizes = [degrees[node] * 10 for node in G_viz.nodes()]
        G_undirected = G_viz.to_undirected()
        partition = community_louvain.best_partition(G_undirected)
        node_colors = [partition[node] for node in G_viz.nodes()]

        pos = nx.spring_layout(G_viz,k = 0.3,iterations=50,seed = 42)

        plt.figure(figsize=(14,10))

        nx.draw_networkx_nodes(G_viz,pos,node_size = node_sizes,
                                   node_color = node_colors,cmap= plt.cm.Set3,alpha = 0.8)
        nx.draw_networkx_edges(G_viz,pos,edge_color = 'gray',alpha = 0.2,arrows=False)

        top_influencers = sorted(degrees.items(),key = lambda x: x[1],reverse = True) [:5]
        labels = {node: node for node , _ in top_influencers}
        nx.draw_networkx_labels(G_viz,pos,labels,font_size = 10)

        plt.title(f"Social Network Analysis\n" f"Nodes {len(G_viz)} , "
                      f"Edges: {G_viz.number_of_edges()}, Communities: {len(set(partition.values()))}")
        plt.axis('off')
        plt.text(0.95,0.02,f"Node size = Degree\n Color = Community",
                     transform = plt.gca().transAxes,fontsize = 10, verticalalignment='top',horizontalalignment='right',
                     bbox  =dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))
        plt.tight_layout()
        plt.savefig("data/network_visualization.png", dpi = 600,bbox_inches = 'tight')
        plt.show()

    def generate_network_report(self):
        if self.G is None:
            raise ValueError("Network not loaded")
        print("Creating network visualization")
        metrics = self.calculate_metrics()

        report = {
            'Network_Summary':{
                'Total Users':metrics['num_nodes'],
                'Total Connections': metrics['num_edges'],
                'Network Density': f"{metrics['density']:.4f}",
                'Average Degree' : f"{metrics['avg_degree']:.2f}",
                'Number of Communities': f"{metrics['num_communities']}",
                'Average Clustering': f"{metrics['avg_clustering']:.4f}"
            },
            'Top_Influencers':
                [
                    {'user' : user, 'followers': deg}
                    for user, deg in metrics['Top_Influencers'][:5]
                ],
            'Degree_Distribution': dict(Counter(dict(self.G.degree()).values())),
            'Centrality_Analysis':{
                'Most Central by Degree': max(metrics['degree_centrality'].items(),
                                              key = lambda x:x[1])[0],
                'Most Central by Betweenness': max(metrics['betweenness_centrality'].items(),
                                                   key = lambda x:x[1])[0],
                'Most Central by Closeness': max(metrics['closeness_centrality'].items(),
                                                 key = lambda x:x[1])[0],
            }
        }
        print("Network Analysis report".upper())
        print("-"*30)
        for key,value in report['Network_Summary'].items():
            print(f"{key} : {value}")
        print("Top 5 influencers:".upper())
        for i , inf in enumerate(report['Top_Influencers'],1):
            print(f"{i}.{inf['user']}(followers: {inf['followers']})")
        print("Centrality Analysis:".upper())
        for centrality,user in report['Centrality_Analysis'].items():
            print(f"{centrality} : {user}")
        self._plot_degree_distribution(report['Degree_Distribution'])
        return report

    def _plot_degree_distribution(self,degree_dist):
        degrees = list(degree_dist.keys())
        counts = list(degree_dist.values())

        plt.figure(figsize = (10,6))
        plt.bar(degrees,counts,alpha= 0.7,edgecolor= 'black')
        plt.xlabel('Number of Connections')
        plt.ylabel('Number of Users')
        plt.title('Degree Distribution by Social Network')
        plt.grid(True,alpha=0.3)

        if max(degrees) > 10:
            inset = plt.axes([0.6,0.6,0.3,0.3])
            inset.loglog(degrees,counts,'bo',alpha = 0.6)
            inset.set_xlabel("Degree (log)")
            inset.set_ylabel('Count (log)')
            inset.set_title('Log-Log Plot')
            inset.grid(True,alpha=0.3)

        plt.tight_layout()
        plt.savefig("data/degree-distribution.png", dpi = 600,bbox_inches = 'tight')
        plt.show()

        print("Saved Degree Distribution plot")
if __name__ == '__main__':
    analyzer = NetworkAnalyzer()













