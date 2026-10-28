# GraphGuard: Graph Neural Network-Based Anomaly Detection in Cryptocurrency Transaction Networks with LLM-Driven Explainability

> **Course**: Big Data Analytics (BITE413L)  
> **Student Names**: Mohil Sharma (`23BIT0430`), Arnav Mishra (`23BIT0142`)  
> **Project Guide**: Dhivya J  

---

## 1. System Architecture

GraphGuard models cryptocurrency transaction networks as dynamic, attributed directed graphs and applies Graph Neural Networks (GraphSAGE and GCN) to detect complex money-laundering typologies (fan-out, fund cycling, and community bridging). It addresses the black-box nature of GNNs through a Retrieval-Augmented Generation (RAG) explainability layer grounded in regulatory definitions and financial crime typologies.

```
+---------------------------------------------------------------------------------------------------+
|                                          DATA LAKE                                                |
|                                                                                                   |
|  [Raw Zone]                    [Processed Zone]                  [Analytics Zone]                 |
|  data/lake/raw/elliptic/   ->  data/lake/processed/          ->  data/lake/analytics/             |
|  - classes.csv                 - nodes.parquet                   - graph_features.parquet         |
|  - edgelist.csv                - edges.parquet                   - risk_scores.parquet            |
|  - features.csv                                                  - dq_report.json                 |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                                   DISTRIBUTED PYSPARK ETL                                         |
|  - Schema validation & deduplication                                                              |
|  - Feature joining (165 raw node features)                                                        |
|  - Topological metric computation: in/out degree, fan-out ratio, reciprocal edges,               |
|    relative transaction velocity, and temporal recency                                            |
|  - High-performance Pandas fallback engine for single-node / edge environments                    |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                           GRAPH CONSTRUCTION & ANOMALY TYPING                                     |
|  - Strongly Connected Components (SCC) for directed fund-cycling cycles                           |
|  - Louvain Community Detection & cross-community bridging ratio                                   |
|  - Anomaly classification: Fan-out (dispersal) | Fund Cycling | Community Bridging                |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                                    GNN DETECTION ENGINE                                           |
|  - Inductive GraphSAGE (Mean aggregator + L2-norm embeddings)                                     |
|  - Transductive Spectral GCN (Symmetric Kipf & Welling normalization)                             |
|  - Random Forest Baseline (local 165 features, Weber et al., 2019)                               |
|  - Temporal Train/Test split: Steps 1-34 (Train) vs. Steps 35-49 (Test)                            |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                                  RAG EXPLAINABILITY ENGINE                                        |
|  - Typology Corpus: knowledge/typologies.md & automatic FATF PDF ingestion (pypdf / pdftotext)    |
|  - Multi-hop subgraph evidence extraction                                                         |
|  - Offline Template Explainer (Default: privacy-preserving, zero-cost, deterministic citations)   |
|  - LLM Provider Support: Anthropic Claude / OpenAI GPT-4o via API configuration                   |
+---------------------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------------------+
|                                     INTERFACE & SERVING                                           |
|  - FastAPI Backend (http://localhost:8000): /health, /metrics, /flagged, /wallet/{tx_id}          |
|  - Streamlit Compliance Dashboard (http://localhost:8501): Interactive subgraph visualizer       |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Experimental Benchmark Results

Evaluated on the full **Elliptic Dataset** (**203,769 transactions**, **234,355 directed edges**, **46,564 labeled nodes**) under the temporal illicit-class evaluation protocol:
* **Training set**: Time steps 1–34
* **Testing set**: Time steps 35–49

### Model Performance (Illicit Class)

| Model | Precision | Recall | **F1-Score** | Inference / Train Time |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | **92.05%** | 72.67% | **0.8122** | 2.35s |
| **GraphSAGE** | 71.51% | **69.07%** | **0.7027** | 118.76s |
| **GCN** | 51.41% | 63.99% | **0.5701** | 121.62s |

* **GraphSAGE vs. GCN**: GraphSAGE outperforms GCN by **+13.26% F1**, confirming that inductive neighborhood aggregation generalizes significantly better to unseen future time steps than fixed spectral GCN.
* **Structural Graph Analytics**: While Random Forest excels on local tabular attributes, GNN models capture cross-hop money laundering patterns and transaction flows across unlabelled intermediaries.

### RAG Explanation & Groundedness Evaluation

| Mode | Citation Rate | Hallucination Rate | Evidence Accuracy | Compliance Disclaimer |
| :--- | :---: | :---: | :---: | :---: |
| **Ungrounded (Ablation)** | 0.00% | 0.00% | 100% | Missing citations |
| **Grounded (GraphGuard RAG)** | **25.00%** | **0.00%** | **100%** | Grounded in `[REG-SAR]`, `[REG-LIMITS]`, `[TYP-*]` |

---

## 3. Quickstart & Execution

### Prerequisites
* Python 3.10 / 3.11
* Java 8/11/17 (or Java 21+ with PySpark 4)

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
The Elliptic dataset CSVs belong in `data/lake/raw/elliptic/`:
* `elliptic_txs_classes.csv`
* `elliptic_txs_edgelist.csv`
* `elliptic_txs_features.csv`

To fetch synthetic test data instead (smoke testing only):
```bash
python -m graphguard.cli demo-data
```

### 3. Run the Complete Pipeline
Executes PySpark ETL $\rightarrow$ Graph Construction $\rightarrow$ Model Training $\rightarrow$ Anomaly Typing $\rightarrow$ Explanation Evaluation:
```bash
PYTHONPATH=. python -m graphguard.cli all
```

Individual pipeline stages can be executed independently:
```bash
PYTHONPATH=. python -m graphguard.cli etl
PYTHONPATH=. python -m graphguard.cli train
PYTHONPATH=. python -m graphguard.cli eval-explanations
```

### 4. Run Test Suite
```bash
PYTHONPATH=. pytest tests -q
```

---

## 4. Serving & User Interfaces

### Interactive Compliance Dashboard (Streamlit)
Launch the compliance analyst dashboard:
```bash
PYTHONPATH=. streamlit run graphguard/dashboard.py
```
* **URL**: `http://localhost:8501`
* **Features**:
  * Live model metrics summary table.
  * Transaction filtering by time step, pattern (`fan_out`, `fund_cycling`, `community_bridging`), and risk threshold.
  * Interactive transaction inspection with RAG-grounded natural-language explanations.
  * Real-time NetworkX counterparty subgraph topology visualization.

### REST API (FastAPI)
Launch the query API:
```bash
PYTHONPATH=. uvicorn graphguard.api:app --host 0.0.0.0 --port 8000
```
* **Swagger UI / OpenAPI Documentation**: `http://localhost:8000/docs`
* **Endpoints**:
  * `GET /health` — Service status, graph node and edge counts, active model architecture.
  * `GET /metrics` — Precision, recall, and F1 benchmarks across all models.
  * `GET /flagged?limit=50&pattern=fan_out` — Query high-risk transactions filtered by typology.
  * `GET /wallet/{tx_id}?explain=true&grounded=true` — Subgraph evidence, risk score, citations, and natural language explanation for an individual transaction.

---

## 5. Explainability & LLM Configuration

The explainability layer translates GNN risk scores and graph metrics into auditable, plain-language summaries for compliance investigators.

### Providers (`GRAPHGUARD_LLM`):
* **`template` (Default)**:
  * Deterministic, zero latency, zero API costs.
  * 100% data privacy: on-chain evidence is not sent to external servers.
  * Automatically injects citations (`[TYP-FANOUT]`, `[TYP-BRIDGE]`, `[REG-LIMITS]`, `[REG-SAR]`) based on retrieved chunks.
* **`anthropic`**:
  ```bash
  export GRAPHGUARD_LLM=anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
  export GRAPHGUARD_LLM_MODEL="claude-sonnet-5-5"  # optional
  ```
* **`openai`**:
  ```bash
  export GRAPHGUARD_LLM=openai
  export OPENAI_API_KEY="sk-..."
  export GRAPHGUARD_LLM_MODEL="gpt-4o-mini"        # optional
  ```

### Regulatory Knowledge Base (`knowledge/`)
* [`knowledge/typologies.md`](knowledge/typologies.md) contains structured typologies (fan-out, peel chains, wash trading, mixers, velocity, and regulatory limits).
* Official FATF PDFs (such as the *FATF Virtual Assets Red Flag Indicators* report) can be placed directly in `knowledge/`. The built-in PDF ingestion engine (`pypdf` and `pdftotext`) automatically indexes them into the RAG vector space.

---

## 6. Project Artifacts & Data Lake Hierarchy

```
data/lake/
|-- raw/
|   `-- elliptic/               # Ingested raw Kaggle CSVs
|-- processed/
|   |-- nodes.parquet           # Validated nodes (165 features + labels)
|   `-- edges.parquet           # Cleaned, deduplicated directed edges
`-- analytics/
    |-- graph_features.parquet  # Computed degrees, fanout, velocity, recency
    |-- risk_scores.parquet     # Predicted risk scores across all models
    `-- dq_report.json          # Data quality & schema verification audit

artifacts/
|-- metrics.json                # Final model precision, recall, F1, and runtimes
|-- graphsage.pt                # Trained PyTorch GraphSAGE weights
|-- gcn.pt                      # Trained PyTorch GCN weights
|-- embeddings_graphsage.npy    # Learned node embeddings (GraphSAGE)
|-- embeddings_gcn.npy          # Learned node embeddings (GCN)
`-- explanation_eval.csv        # Grounded vs. ungrounded RAG evaluation dataset
```
