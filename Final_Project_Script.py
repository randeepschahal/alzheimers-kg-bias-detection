import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from pyvis.network import Network
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk import bigrams
import re
from wordcloud import WordCloud

# XML Parser
def parse_xml_to_graph(xml_file, graph=None):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    G = graph if graph else nx.DiGraph()

    # Add entities - try both uppercase and lowercase
    for entity in root.findall('./Entities/*'):
        entity_id = entity.attrib.get('id')
        if entity_id and entity_id not in G.nodes:
            entity_type = entity.attrib.get('type', entity.tag)
            
            # Try to get name from attribute first
            entity_name = entity.attrib.get('name')
            
            # If no name attribute, try to get text content from the entity
            if not entity_name or entity_name.strip() == "":
                entity_name = entity.text
            
            # If still no name, use the ID-based fallback
            if not entity_name or entity_name.strip() == "":
                entity_name = f"{entity_type}_{entity_id}"
            
            G.add_node(entity_id, type=entity_type, name=entity_name.strip() if isinstance(entity_name, str) else entity_name)

    # Add relations - handle both uppercase and lowercase, and nested structure
    # First try lowercase 'relation'
    for relation in root.findall('.//relation'):
        # Try attribute-based (lowercase)
        source = relation.attrib.get('source') or relation.attrib.get('subject')
        target = relation.attrib.get('target') or relation.attrib.get('object')
        rel_type = relation.attrib.get('type') or relation.attrib.get('predicate')

        # If missing, try child tags
        if source is None:
            src_elem = relation.find('./Source') or relation.find('./source')
            if src_elem is not None:
                source = src_elem.attrib.get('ref') or src_elem.text
        if target is None:
            tgt_elem = relation.find('./Target') or relation.find('./target')
            if tgt_elem is not None:
                target = tgt_elem.attrib.get('ref') or tgt_elem.text

        # Skip invalid relations
        if source is None or target is None:
            print(f"Skipping invalid relation in {xml_file}: source={source}, target={target}")
            continue

        # Only add edge if both nodes exist
        if source in G.nodes and target in G.nodes:
            if not G.has_edge(source, target):
                G.add_edge(source, target, predicate=rel_type)
        else:
            print(f"Skipping relation {source} -> {target}: nodes not found in graph")

    # Also try uppercase 'Relation' for backward compatibility
    for relation in root.findall('.//Relation'):
        # Try attribute-based (uppercase)
        source = relation.attrib.get('source') or relation.attrib.get('subject')
        target = relation.attrib.get('target') or relation.attrib.get('object')
        rel_type = relation.attrib.get('type') or relation.attrib.get('predicate')

        # If missing, try child tags
        if source is None:
            src_elem = relation.find('./Source') or relation.find('./source')
            if src_elem is not None:
                source = src_elem.attrib.get('ref') or src_elem.text
        if target is None:
            tgt_elem = relation.find('./Target') or relation.find('./target')
            if tgt_elem is not None:
                target = tgt_elem.attrib.get('ref') or tgt_elem.text

        # Skip invalid relations
        if source is None or target is None:
            print(f"Skipping invalid relation in {xml_file}: source={source}, target={target}")
            continue

        # Only add edge if both nodes exist
        if source in G.nodes and target in G.nodes:
            if not G.has_edge(source, target):
                G.add_edge(source, target, predicate=rel_type)
        else:
            print(f"Skipping relation {source} -> {target}: nodes not found in graph")

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

# PyVis Knowledge Graph Visualization
def visualize_graph_interactive(G, output_file="KG.html", remove_isolated=True):
    # Remove isolated nodes
    if remove_isolated:
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            print(f"Removing {len(isolated_nodes)} isolated nodes")
            G.remove_nodes_from(isolated_nodes)
    
    # Check if graph has nodes
    if len(G.nodes) == 0:
        print("Warning: Graph has no nodes!")
        return
    
    print(f"Graph stats: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
    
    # Map node types to colors
    type_colors = {
        "CognitiveTest": "skyblue", 
        "ClinicalCategory": "lightgreen", 
        "Clinical_Trial": "lightgreen",
        "Biomarker": "orange", 
        "Pathology": "gray", 
        "Pathway": "gray",
        "Pathological_Feature": "gray",
        "Imaging": "indigo", 
        "MRI_Feature": "purple", 
        "Test": "violet", 
        "Genetic": "tomato", 
        "RiskScore": "darksalmon", 
        "RiskFactor": "darksalmon", 
        "Disease": "red", 
        "Disease_Stage": "red",
        "Condition": "red", 
        "Drug": "green", 
        "Organization": "blue", 
        "Gene": "darkkhaki", 
        "Cell": "goldenrod",
        "Protein": "darkgoldenrod", 
        "Molecular_Process": "black", 
        "Receptor": "brown", 
        "Signaling_Pathway": "brown",
        "Structure": "pink", 
        "Classification": "tan"
    }
   
    # Add nodes
    for node, data in G.nodes(data=True):
        node_name = str(data.get("name", node))
        node_type = data.get("type", "Unknown")
        color = type_colors.get(node_type, "gray")
        net.add_node(str(node), label=node_name, title=f"Type: {node_type}", color=color)
    
    # Add edges
    for u, v, data in G.edges(data=True):
        predicate = data.get("predicate", "")
        net.add_edge(str(u), str(v), title=predicate)
    
    # Enable physics for better layout
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4)
    
    # Save interactive HTML
    net.write_html(output_file)
    print(f"Interactive graph saved as {output_file}")

# GNN for Bias Label Calculation 
def generate_bias_labels(G):
    labels = {}
    degrees = np.array([G.degree(n) for n in G.nodes()])
    low, high = np.percentile(degrees, [30, 70])

    for n in G.nodes():
        d = G.degree(n)
        if d < low:
            labels[n] = 0   # underrepresented
        elif d > high:
            labels[n] = 2   # overemphasized
        else:
            labels[n] = 1   # normal
    return labels

def kg_to_gnn_data(G, bias_labels):
    type_mapping = {"CognitiveTest":0, "ClinicalCategory":1, "Clinical_Trial":1, "Biomarker":2, "Pathology":3, "Pathway":3, "Pathological_Feature":3, "Imaging":4, "MRI_Feature":4, 
    "Test":4, "Genetic":5, "RiskScore":5, "RiskFactor":5, "Disease":6, "Disease_Stage":6, "Condition":6, "Drug":7, "Organization":8, "Gene":9, 
    "Cell":10, "Protein":10, "Molecular_Process":11, "Receptor": 12, "Signaling_Pathway":12, "Structure":13, "Classification":14, "Unknown":15}
    num_types = len(type_mapping)

    node_features = []
    node_labels = []
    node_id_map = {}

    for idx, (node, data) in enumerate(G.nodes(data=True)):
        node_id_map[node] = idx
        
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg

        node_type = data.get("type", "Unknown")
        type_id = type_mapping.get(node_type, type_mapping["Unknown"])
        one_hot_type = [1 if i==type_id else 0 for i in range(num_types)]

        features = [in_deg, out_deg, total_deg] + one_hot_type
        node_features.append(features)

        node_labels.append(bias_labels[node])  # <-- now using trust/bias label

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)

    edges = [[node_id_map[u], node_id_map[v]] for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h)

def train_gnn(data):
    model = GNN(in_dim=data.x.size(1), hidden_dim=32, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model
def compute_predicted_bias(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds, probs

# Textual Indicator Analysis
ad_txt = Path("AD_Merged.txt")    
stopwords_path = Path("stopwords_en.txt")

with ad_txt.open(encoding="utf-8") as f:
    ad_raw = f.read()
with stopwords_path.open(encoding="utf-8") as f:
    stopwords = set(w.strip().lower() for w in f if w.strip())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    issue_words = {"alzheimer", "alzheimer's", "disease", "ad"}
    tokens = re.findall(r"[a-z']{3,}", text) 
    tokens = [t for t in tokens if t not in stopwords and t not in issue_words]
    return tokens

tokens_ad = clean_text(ad_raw)

# Conducting Sentiment Analysis 
analyzer = SentimentIntensityAnalyzer()
ad_sent = analyzer.polarity_scores(" ".join(tokens_ad))

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

lex_div_ad = lexical_diversity(tokens_ad)

# Calculate Word Frequency and Extract Bigrams 
freqs_ad = Counter(tokens_ad)
bigrams_ad = Counter(list(bigrams(tokens_ad)))

# Word Cloud
def make_wordcloud(freqs, title, filename):
    wc = WordCloud(width=1000, height=700, background_color="white").generate_from_frequencies(freqs)
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    folder = Path(".")
    G_combined = nx.DiGraph()

    xml_files = list(folder.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")
    
    for xml_file in xml_files:
        print(f"Processing {xml_file}")
        G_combined = parse_xml_to_graph(xml_file, G_combined)
    
    df_features = compute_graph_features(G_combined)
    print("\n=== GRAPH FEATURES ===")
    print(df_features)
    df_features.to_csv("graph_features.csv", index=False)

    print(f"\nTotal nodes: {len(G_combined.nodes)}")
    print(f"Total edges: {len(G_combined.edges)}")

    visualize_graph_interactive(G_combined, output_file="Interactive_AD_KG.html")

    bias_labels = generate_bias_labels(G_combined)
    pd.DataFrame([
        {"node": n, "name": G_combined.nodes[n].get("name", ""), "bias_label": bias_labels[n]}
        for n in G_combined.nodes()
    ]).to_csv("initial_bias_labels.csv", index=False)
    print("\nGenerated bias labels.")

    data = kg_to_gnn_data(G_combined, bias_labels)
    print("Converted KG into PyTorch Geometric Data object.")

    print("\nTraining GNN...")
    model = train_gnn(data)

    preds, probs = compute_predicted_bias(model, data)

    rows = []
    for node, idx in zip(G_combined.nodes(), range(len(G_combined.nodes()))):
        rows.append({
            "node_id": node,
            "name": G_combined.nodes[node].get("name", ""),
            "type": G_combined.nodes[node].get("type", ""),
            "true_bias_label": int(data.y[idx].item()),
            "predicted_bias_label": int(preds[idx].item()),
            "prob_underrepresented": float(probs[idx][0].item()),
            "prob_normal": float(probs[idx][1].item()),
            "prob_overemphasized": float(probs[idx][2].item()),
        })

    df_results = pd.DataFrame(rows)
    df_results.to_csv("gnn_bias_predictions.csv", index=False)
    print("gnn_bias_predictions.csv created successfully.")

    print("\n========== TEXTUAL INDICATOR ANALYSIS ==========\n")
    print(f"Total tokens after cleaning: {len(tokens_ad)}")
    print(f"Unique vocabulary size: {len(set(tokens_ad))}")

    print("\n--- Sentiment Scores (VADER) ---")
    print(ad_sent)

    basic_neu, neutrality_index = compute_neutrality(ad_sent)
    print("\n--- Neutrality Analysis ---")
    print(f"Basic Neutrality: {basic_neu:.4f}")
    print(f"Neutrality Index (neu / (pos + neg)): {neutrality_index:.4f}")

    print("\n--- Lexical Diversity ---")
    print(f"Lexical Diversity Ratio: {lex_div_ad:.4f}")

    print("\nTop 20 Most Frequent Words:")
    for word, count in freqs_ad.most_common(20):
        print(f"{word}: {count}")
    
    print("\nTop 20 Most Frequent Bigrams:")
    for bg, count in bigrams_ad.most_common(20):
        print(f"{bg}: {count}")
    
    print("\nGenerating word cloud...")
    make_wordcloud(freqs_ad, "AD Text Word Cloud", "AD_WordCloud.png")
    print("Word cloud saved as AD_WordCloud.png")
