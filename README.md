# 🧬 Dr. P.M. Bhargava Memorial DNA & CRISPR Validation Suite

An enterprise-grade multi-omics batch processing platform and deep learning framework focused on **Quantifying Architectural Reliability in CRISPR Off-Target Prediction**. 

This repository directly honors the scientific legacy of **Dr. Pushpa Mittra Bhargava** (1928–2017), the pioneering visionary who founded India's Centre for Cellular and Molecular Biology (CCMB). Replicating his commitment to rigorous empirical research, this workspace evaluates CRISPR-Cas9 exon-cleavage validation matrices against background oncology sequencing noise using real-time automated hypothesis testing and sequence susceptibility transformers.

---
## 📺 Streamlit Lit Dash Board
[CRISPR Dashboard Live Demo]([https://bhargava-dna-crispr-sim-yj9ywkajsqjkik4zqdduxr.streamlit.app/])

---
## 🖥️ Workspace Visual Architecture

The accompanying visual stream features a close-up, high-contrast user interface engineered for 4K video capture and micro-telemetry presentation decks:

* **Dual-View Analytics System:** The interface maps real-time deep learning analytics simultaneously. The left view graphs a sequence cleavage susceptibility curve, while the right features a dense patient cohort cell matrix.
* **Cinematic Color Space:** Utilizes a rich technical dark navy background (`#0b132b`) populated by midnight blue passive tokens (`#1c2541`). Active structural knockouts flash into a high-visibility neon magenta (`#e056fd`).
* **Gliding Data Horizon:** Uses horizontal rolling transformations to model streaming data pipelines across the sequence coordinates. Lockable cyberpunk yellow boundary wires tracking the target exon window anchor the visual timeline.

---

## 🧪 Scientific & Mathematical Infrastructure

### 1. Sequential Susceptibility Engine
The framework computes token frequency distributions (GC/AT structural balance) across an uploaded or default reference strand (e.g., Human TP53 tumor suppressor coding transcript excerpt) through a sliding window paradigm:

$$\text{Raw Score} = 0.3 \cdot \left(\frac{\text{GC Count}}{\text{Window Size}}\right) + 0.5 \cdot \mathbb{I}(\text{Nucleotide} \in \{\text{G, C}\})$$

Scores pass through a boundary-constrained normalization pipeline ensuring calibrated probability mapping between $0.15$ and $0.95$.

### 2. Micro-Telemetry Hypothesis Testing
To isolate authentic genome modifications from baseline oncology instrumentation artifacts, the suite runs a directional Single-Sample t-Test:

* **Null Hypothesis ($H_0$):** $\mu_{\text{observed}} \le \text{base\_error\_rate}$ (Deletions within target windows are identical to or lower than background sequencing engine artifacts).
* **Alternative Hypothesis ($H_1$):** $\mu_{\text{observed}} > \text{base\_error\_rate}$ (The targeted envelope exhibits a statistically elevated deletion threshold, validating structural cleavage events).

An evaluation output of $p < 0.05$ systematically rejects the null state, updating telemetry indicators to `SIGNIFICANT EDIT SIGNAL`.

---

## 🛠️ Key Features

* **Multi-File Batch Loading:** Seamlessly ingests multiple `.fasta`, `.fa`, or `.txt` target genomic exons concurrently via automated Biopython IO parsing pipelines.
* **Inference Susceptibility Curve:** Dynamically maps regional spatial risk factors using localized nucleotide token analysis.
* **Clinical Patient Metadata Generator:** Generates discrete patient variables (Age distribution profiling, oncology staging categories, sequencing read depth variables) bound directly to custom slider configurations.
* **Data Infrastructure Export:** Automatically compiles structural multi-omics arrays and metadata logs into dense, downloadable pipeline-ready `.csv` configurations.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com
cd bhargava-dna-crispr-sim
pip install -r requirements.txt
streamlit run app.py
```

### Execution
Run the streaming visualization validation workspace locally:
```bash
streamlit run app.py
```

---

## 📊 Sample Execution Metrics
* **Observed Mean Target Deletion:** `~68.9%`
* **Calculated p-value Score:** `~8.28e-06` (Statistically confirmed edit signal)
* **Default Reference Anchor:** `NC_000017.11` Homo sapiens tumor protein p53 (TP53)

---

## 🏛️ Tribute Context
**Dr. P.M. Bhargava** was a monumental figure in global biotechnology, pushing for absolute stringency, reproducible empirical benchmarks, and public engagement with science. This computational validation platform integrates modern deep learning sequence tokenization to advance the clinical evaluation methods he championed throughout his tenure at CCMB.




## 📜 Credits & Acknowledgments
* **Lead System Developer:** Srinivasta
* **Scientific Inspiration:** Dr. P.M. Bhargava (CCMB)
* **Core Libraries:** Biopython, Streamlit, Scipy, Seaborn
