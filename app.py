import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from scipy import stats

# Establish screen parameters and workspace formatting templates
st.set_page_config(
    page_title="Dr. P.M. Bhargava DNA Validation Suite", 
    page_icon="🧬", 
    layout="wide"
)

# Custom dark-mode styling overrides to isolate boundaries during screen recording
st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 0rem;}
    h1 {color: #e056fd; font-weight: 800; font-family: 'Courier New', monospace;}
    .reportview-container .main .block-container{ max-width: 95%; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------
# MAIN HEADER BLOCK: Track 1 Dedication & Scientific Tribute
# ----------------------------------------------------------------------------------------------------
st.title("🧬 Dr. P.M. Bhargava Memorial DNA & CRISPR Validation Suite")
st.markdown("""
    **Track 1: Bioinformatics and DNA Analysis**  
    *This interactive suite directly honors the scientific legacy of **Dr. Pushpa Mittra Bhargava** (1928–2017), 
    the pioneering visionary who founded India's Centre for Cellular and Molecular Biology (CCMB). Replicating his commitment 
    to rigorous empirical research, this workspace evaluates CRISPR-Cas9 exon-cleavage validation matrices 
    against background oncology sequencing noise using real-time automated hypothesis testing modeling.*
""")
st.markdown("---")

# ----------------------------------------------------------------------------------------------------
# BLOCK 1: SIMULATOR VARIABLES & PHYSICS CONTROLS (Sidebar Engine UI)
# ----------------------------------------------------------------------------------------------------
st.sidebar.header("🎛️ Physics & Simulation Controls")
st.sidebar.markdown("*Adjust variables to observe real-time variant transformations.*")

n_patients = st.sidebar.slider("Patient Cohort Size (Data Rows)", min_value=5, max_value=40, value=15, step=1)
crispr_efficiency = st.sidebar.slider("Expected CRISPR Success Rate (%)", min_value=0, max_value=100, value=80, step=5) / 100
base_error_rate = st.sidebar.slider("Baseline Sequencing Noise (%)", min_value=0, max_value=15, value=4, step=1) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Target Exon Window")
target_start = st.sidebar.number_input("CRISPR Bound Start (1-indexed base)", min_value=1, max_value=20, value=12) - 1
target_end = st.sidebar.number_input("CRISPR Bound End (1-indexed base)", min_value=2, max_value=30, value=18) - 1

# Structural validation guard rails
if target_start >= target_end:
    st.sidebar.error("Coordinate Error: Start position must sit lower than end position.")
    st.stop()

# ----------------------------------------------------------------------------------------------------
# BLOCK 2: SEQUENCING CONTEXT & LIVE BIOPYTHON IO PIPELINE
# ----------------------------------------------------------------------------------------------------
st.subheader("📂 Step 1: Input Genomic FASTA Data Context")

# Built-in clinical target: Human TP53 tumor suppressor coding transcript excerpt
default_fasta = """>NC_000017.11:c7579699-7579670 Homo sapiens tumor protein p53 (TP53), mRNA Exon Excerpt
ATGGAGGAGCCGCAGTCAGATCCTAGCGTC"""

uploaded_file = st.file_uploader("Upload Target Exon Record (.fasta, .fa, .txt)", type=["fasta", "fa", "txt"])

if uploaded_file is not None:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    fasta_content = stringio.read()
    st.success("Custom genomic string mounted successfully.")
else:
    fasta_content = default_fasta
    st.info("ℹ️ Using default integrated TP53 reference sequence to anchor diagnostics.")

# Execute background Biopython parsing sequence
fasta_io = io.StringIO(fasta_content)
try:
    record = SeqIO.read(fasta_io, "fasta")
    ref_seq = record.seq
    n_bases = len(ref_seq)
except Exception as e:
    st.error(f"IO Parsing Exception: {e}. Confirm standard FASTA spacing protocols.")
    st.stop()

if target_end > n_bases:
    st.error(f"Coordinate Overflow: Selected target boundary ({target_end+1}bp) exceeds gene segment scale ({n_bases}bp).")
    st.stop()

# Print dynamic transcription analytics to screen containers
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"🧬 **Header Description:** `{record.description}`")
    st.markdown(f"📏 **Genome Window Length:** `{n_bases} base pairs`")
with col2:
    st.markdown(f"🔤 **Reference Exon Nucleotides:** `{ref_seq}`")
    st.markdown(f"🧪 **Translated Peptide Chain:** `{ref_seq.translate()}`")

st.markdown("---")

# ----------------------------------------------------------------------------------------------------
# BLOCK 3: CLINICAL TELEMETRY DASHBOARD & HIGH-CONTRAST HEATMAP ENGINE
# ----------------------------------------------------------------------------------------------------
st.subheader("📊 Step 2: Cohort Variant Deletion Matrix & Peak Telemetry")

# Initialize seed for consistent, deterministic visual simulation capturing
np.random.seed(42)
mutation_matrix = np.zeros((n_patients, n_bases))
crispr_success_mask = np.random.rand(n_patients) < crispr_efficiency

# Build variant profiling states across the sample rows
for patient_idx in range(n_patients):
    for pos in range(n_bases):
        if crispr_success_mask[patient_idx] and (target_start <= pos < target_end):
            # Target range knockdown: 85% real mutation likelihood vs 15% missed cuts
            mutation_matrix[patient_idx, pos] = np.random.choice([1, 0], p=[0.85, 0.15])
        else:
            # Out-of-bounds metrics: Regulated by standard baseline sequencing noise
            mutation_matrix[patient_idx, pos] = np.random.choice([1, 0], p=[base_error_rate, 1 - base_error_rate])

# Package dynamic arrays into a clear pandas analytical block
patient_ids = [f"Patient_{i+1:02d}" for i in range(n_patients)]
position_labels = [f"{ref_seq[i]}_{i+1}" for i in range(n_bases)]
df_matrix = pd.DataFrame(mutation_matrix, index=patient_ids, columns=position_labels)

# Statistical validation logic: single-sample t-test evaluating target window density
observed_target_rates = df_matrix.iloc[:, target_start:target_end].mean(axis=1)
t_stat, p_value = stats.ttest_1samp(observed_target_rates, base_error_rate, alternative='greater')
mean_deletion_pct = observed_target_rates.mean() * 100

# Draw Peak Telemetry Value Matrix cards
m1, m2, m3 = st.columns(3)
m1.metric("Observed Mean Target Deletion", f"{mean_deletion_pct:.1f}%")
m2.metric("Calculated p-value Score", f"{p_value:.2e}")

with m3:
    st.markdown("**Hypothesis Threshold Status:**")
    if p_value < 0.05:
        st.success("✅ SIGNIFICANT EDIT SIGNAL")
    else:
        st.error("🚨 BACKGROUND NOISE DOMINANT")

# Build the custom Matplotlib rendering canvas using high-visibility neon colors for GIF processing
fig, ax = plt.subplots(figsize=(14, 5.5))
# Matte slate background with hyper-vibrant pink mutation indicator blocks
sns.heatmap(df_matrix, cmap=["#2c3e50", "#e056fd"], cbar=False, linewidths=0.8, linecolor="#34495e", ax=ax)

ax.set_title("Variant Cleavage Matrix Map (Vibrant Pink = Target Base Knocked Out / Absent)", fontsize=12, weight='bold', pad=15)
ax.set_xlabel("Genomic Sequence Base Identifiers & Coordinate Index positions", fontsize=10, labelpad=8)
ax.set_ylabel("Simulated Patient Sample Fields", fontsize=10)

# Drop sharp neon yellow coordinate boundary wirelines over target coordinates
ax.axvline(x=target_start, color='#ffea00', linestyle='--', linewidth=2.5, label="CRISPR Target Envelope Bounds")
ax.axvline(x=target_end, color='#ffea00', linestyle='--', linewidth=2.5)

# Render completed visual element to interface window
st.pyplot(fig)
st.markdown("---")

# ----------------------------------------------------------------------------------------------------
# COMPREHENSIVE SPREADSHEET EXPORT BLOCK
# ----------------------------------------------------------------------------------------------------
st.subheader("💾 Step 3: Streamlit Data Infrastructure Export")
st.markdown("Extract the evaluated model array below into standard formats for external pipeline ingestion steps.")

csv_data = df_matrix.to_csv().encode('utf-8')
st.download_button(
    label="📥 Download Variant Cleavage Matrix Spreadsheet (.csv)",
    data=csv_data,
    file_name="bhargava_crispr_simulation_output.csv",
    mime="text/csv"
)
