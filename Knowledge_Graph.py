import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk import bigrams

## Knowledge Graph Creation and Feature Calculation

# Parsing XML Files
def parse_xml_to_graph(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    G = nx.DiGraph()

    # Add entities
    for entity in root.findall('./Entities/*'):
        entity_id = entity.attrib.get('id')
        entity_type = entity.tag
        entity_name = entity.attrib.get('name')
        G.add_node(entity_id, type=entity_type, name=entity_name)

    # Add relations
    for relation in root.findall('./Relations/Relation'):
        subj = relation.attrib.get('subject')
        obj = relation.attrib.get('object')
        pred = relation.attrib.get('predicate')
        G.add_edge(subj, obj, predicate=pred)

    return G

# Calculating Knowledge Graph Features
def compute_graph_features(G):
    """
    Returns a pandas DataFrame with node features:
    - in_degree
    - out_degree
    - total_degree
    - semantic type
    - local density (egonet density)
    - graph-wide density
    """
    features = []

    global_density = nx.density(G.to_undirected())

    for node in G.nodes():
        node_data = G.nodes[node]

        # Degree-based features
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg

        # Local connection density (ego-network density)
        ego = nx.ego_graph(G.to_undirected(), node)
        local_density = nx.density(ego) if ego.number_of_nodes() > 1 else 0.0

        features.append({
            "node": node,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": total_deg,
            "local_density": local_density,
            "global_density": global_density
        })

    return pd.DataFrame(features)

# Visualizing Knowledge Graph 
def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 12))
    node_labels = {n: G.nodes[n].get('name', str(n)) for n in G.nodes()}
    node_colors = [
    'skyblue' if G.nodes[n].get('type', '') == 'Disease' else 'lightgreen'
    for n in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'predicate')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    xml_file1 = "Alzheimers_Facts_and_Figures_2024.xml"
    xml_file2 = "AD_Diagnosis.xml"
    xml_file3 = "AD_Disease_Modifying_Therapies_2024.xml"
    xml_file4 = "AD_Pathogenesis_Drug_Development_2024.xml" 
    
    G1 = parse_xml_to_graph(xml_file1)
    G2 = parse_xml_to_graph(xml_file2)
    G3 = parse_xml_to_graph(xml_file3)
    G4 = parse_xml_to_graph(xml_file4)
    
    df_features1 = compute_graph_features(G1)
    df_features2 = compute_graph_features(G2)
    df_features3 = compute_graph_features(G3)
    df_features4 = compute_graph_features(G4)

    print("\n=== GRAPH FEATURES FOR ARTICLE 1 ===")
    print(df_features1)

    print("\n=== GRAPH FEATURES FOR ARTICLE 2 ===")
    print(df_features2)

    print("\n=== GRAPH FEATURES FOR ARTICLE 3 ===")
    print(df_features3)

    print("\n=== GRAPH FEATURES FOR ARTICLE 4 ===")
    print(df_features4)

    visualize_graph(G1)
    visualize_graph(G2)
    visualize_graph(G3)
    visualize_graph(G4)

## Textual Indicator Analysis 

# Loading and Cleaning Raw Article Text Files 
a4_path = Path("AD_Drug_Development.txt")    
stopwords_path = Path("stopwords_en.txt")

with a4_path.open(encoding="utf-8") as f:
    a4_raw = f.read()
with stopwords_path.open(encoding="utf-8") as f:
    stopwords = set(w.strip().lower() for w in f if w.strip())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    tokens = re.findall(r"[a-z']{3,}", text)

    issue_words = {"alzheimer", "disease", "ad"}
    tokens = [t for t in tokens if t not in stopwords and t not in issue_words]
    return tokens

tokens_a4 = clean_text(a4_raw)

# Conducting Sentiment Analysis 
analyzer = SentimentIntensityAnalyzer()
a4_sent = analyzer.polarity_scores(" ".join(tokens_a4))

# Calculating Neutrality 
def compute_neutrality(vader_scores):
    pos = vader_scores.get("pos", 0)
    neg = vader_scores.get("neg", 0)
    neu = vader_scores.get("neu", 0)

    basic_neutrality = neu
    neutrality_index = neu / (pos + neg + 1e-6)
    return basic_neutrality, neutrality_index

# Calculate Lexical Diversity Ratio
def lexical_diversity(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

lex_div_a4 = lexical_diversity(tokens_a4)

# Calculate Word Frequency and Extract Bigrams 
freqs_a4 = Counter(tokens_a4)
bigrams_a4 = Counter(list(bigrams(tokens_a4)))