# Detecting Concept Bias in Alzheimer's Disease Literature with a Knowledge Graph and GNN

EM 626 (AI & Machine Learning for Systems), Stevens Institute of Technology. Final project, Fall 2025.

This project looks at four research papers on Alzheimer's disease (AD) and asks which concepts they over-emphasize and which they under-emphasize. Each paper was summarized as structured XML. The pipeline parses those summaries into entities and relations, builds a combined knowledge graph, labels nodes by bias, and trains a graph neural network on those labels. A separate text analysis runs sentiment and lexical analysis on the full article text.

## Pipeline

1. **XML parsing.** Each `*.xml` file has `<Entities>` and `<Relation>` elements. Entities become nodes, typed by category (Biomarker, Drug, Disease_Stage, and so on). Relations become directed edges labeled with their predicate. All four papers merge into one graph.
2. **Graph features.** For each node: in/out/total degree, semantic type, ego-network density and global graph density (`graph_features.csv`).
3. **Interactive visualization.** A pyvis/vis.js graph colored by entity type (`Interactive_AD_KG.html`, which loads its assets from `lib/`).
4. **Bias labels.** Nodes are labeled by degree percentile (`initial_bias_labels.csv`):
   - `0`: under-represented (below the 30th percentile)
   - `1`: normal
   - `2`: over-emphasized (above the 70th percentile)
5. **GNN.** A two-layer GCN (PyTorch Geometric) trained on degree features plus one-hot entity types to predict the bias labels (`gnn_bias_predictions.csv`, with class probabilities).
6. **Textual indicators.** On the merged article text: VADER sentiment, a neutrality index, lexical diversity, top words and bigrams, and a word cloud (`AD_WordCloud.png`).

## Repository contents

| Path | Description |
| --- | --- |
| `Final_Project_Script.py` | Main pipeline: XML to KG to features to bias labels to GNN, plus sentiment/text analysis |
| `*.xml` | Structured summaries of the articles (the KG input) |
| `AD_Diagnosis.txt`, `AD_Disease-Modifying_Therapies.txt`, `AD_Drug_Development.txt` | Full text of the source articles (publicly available) |
| `AD_Merged.txt` | All article text combined, used for the text analysis |
| `stopwords_en.txt` | Stopword list for text cleaning |

## Running

```bash
pip install -r requirements.txt
python Final_Project_Script.py
```

Run the script from the repository root. It reads the XML files from the current directory and writes these outputs there:

- `graph_features.csv`
- `initial_bias_labels.csv`
- `gnn_bias_predictions.csv`
- `Interactive_AD_KG.html`, plus the `lib/` folder it loads its assets from
- `AD_WordCloud.png`
