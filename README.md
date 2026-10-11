# Detecting Concept Bias in Alzheimer's Disease Literature with a Knowledge Graph and GNN

EM 626 (AI & Machine Learning for Systems), Stevens Institute of Technology. Final project, Fall 2025.

This project looks at four research papers on Alzheimer's disease (AD) and asks which concepts they over-emphasize and which they under-emphasize. Each paper was summarized as structured XML. The pipeline parses those summaries into entities and relations, builds a combined knowledge graph, labels nodes by bias, and trains a graph neural network on those labels. A separate text analysis runs sentiment and lexical analysis on the full article text.

## Pipeline

![Project pipeline](<Project Pipeline (1).png>)

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
| `Knowledge_Graph.py` | Earlier, simpler version of the KG and text-analysis script |
| `AD_Diagnosis.txt`, `AD_Disease-Modifying_Therapies.txt`, `AD_Drug_Development.txt` | Full text of the source articles |
| `AD_Merged.txt` | All article text combined, used for the text analysis |
| `*.xml` | Structured summaries of the articles (the KG input) |
| `stopwords_en.txt` | Stopword list for text cleaning |
| `graph_features.csv` | Per-node graph features |
| `initial_bias_labels.csv` | Degree-based bias labels |
| `gnn_bias_predictions.csv` | GNN predictions and class probabilities |
| `Combined_AD_KnowledgeGraph.graphml` | Combined knowledge graph in GraphML format |
| `Interactive_AD_KG.html`, `lib/` | Interactive graph visualization and its JS/CSS assets |
| `AD_WordCloud.png` | Word cloud of the merged text |
| `Final_Project_Report.pdf`, `Presentation1.pptx` | Final report and presentation |
| `Midterm Report.pdf`, `Midterm Report.docx` | Midterm report |

## Running

```bash
pip install -r requirements.txt
python Final_Project_Script.py
```

Note: the `__main__` block in `Final_Project_Script.py` uses a hard-coded absolute path for the project folder. Change it to `Path(".")` (or your own path) before running the script on a different machine.
