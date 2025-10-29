# 🚀 Transformer-Based Security Framework for Detection of Advanced Persistent Threats (APTs)

This repository contains the official implementation of our research work  
**"Transformer-Based Security Framework for Detection of Advanced Persistent Threats"**.

---

## 🧠 Overview

Advanced Persistent Threats (APTs) are among the most stealthy and multi-staged forms of cyberattacks, often unfolding gradually through reconnaissance, initial compromise, lateral movement, and data exfiltration.  
Traditional intrusion detection systems (IDS) struggle to recognize these subtle transitions between attack phases.

This project proposes a **Transformer-based deep learning framework** for **stage-wise APT detection** using the **Unraveled dataset**.  
Unlike conventional RNN or CNN-based approaches, our Transformer leverages **multi-head self-attention** to capture long-range dependencies and contextual relationships among network features — enabling accurate and fine-grained identification of APT stages.

---

## ⚙️ Key Features

- 🔍 **Stage-wise APT Detection**  
  Detects multiple APT phases: *Benign*, *Reconnaissance*, *Establish Foothold*, *Lateral Movement*, and *Data Exfiltration*.

- 🧩 **Transformer-based Model Architecture**  
  Employs dense embeddings, positional encoding, and multi-head self-attention to learn inter-feature dependencies in tabular network data.

- ⚖️ **Class-Weighted Training**  
  Handles dataset imbalance without synthetic oversampling — preserving the authenticity of the Unraveled dataset.

- 📊 **Comprehensive Evaluation**  
  Evaluated under both **multi-class** and **binary** configurations, achieving near-perfect accuracy and balanced performance.

---

## 🧩 Model Architecture

Input Features (89)
│
Dense Embedding → Positional Encoding
│
Transformer Block (Multi-Head Self-Attention + Feed Forward)
│
Global Average Pooling → Dense Layer + Dropout
│
Softmax Output (5 Classes)

## 📁 Dataset Description

**Dataset Link:** [Unraveled Dataset (GitLab)](https://gitlab.com/asu22/unraveled/-/tree/master/data?ref_type=heads)

The model is trained and evaluated on the **Unraveled dataset** [Myneni et al., 2023],  
a large-scale semi-synthetic APT dataset containing realistic enterprise network traffic data.

- Total records: ~6.8 million flows  
- Duration: 6 weeks  
- Features: 89 (after preprocessing)  
- Attack stages: Benign, Reconnaissance, Establish Foothold, Lateral Movement, Data Exfiltration

### ✂️ Dropped Columns
The following columns were removed due to redundancy, sparsity, or non-generalizable identifiers:
"src_ip, dst_ip, src_mac, dst_mac, requested_server_name, user_agent, Signature, content_type, client_fingerprint, server_fingerprint"

### 🔢 Encoded Columns
Categorical features such as `src_oui`, `dst_oui`, `application_name`,  
`application_category_name`, `Activity`, and `DefenderResponse` were label-encoded.


