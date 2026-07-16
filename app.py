import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from scipy import stats

# 1. EMULATE CINEMATIC 4K HIGH-CONTRAST CANVAS
st.set_page_config(
    page_title="Dr. P.M. Bhargava DNA Validation Suite", 
    page_icon="🧬", 
    layout="wide"
)

# Dark navy digital canvas theme overrides matching your cinematic video profile
st.markdown("""
    <style>
    /* Dark Navy Canvas Background */
    .stApp { background-color: #0b132b; color: #e0e1dd; }
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    
    /* Neon Pink/Magenta Accents for CRISPR Targets */
    h1 { color: #e056fd; font-weight: 800; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(224, 86, 253, 0.3); }
    h2, h3 { color: #48dbfb; font-family: 'Courier New', monospace; }
    
    /* Technical minimalist card formatting */
    div[data-testid="stMetricValue"] { color: #fffa65 !important; font-family: 'Courier New', monospace; font-size: 2rem !important; }
    .stMarkdown { font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# MAIN HEADER BLOCK: Track 1 Dedication & Scientific Tribute
st.title("🧬 Dr. P.M. Bhargava Memorial DNA & CRISPR Validation Suite")
st.markdown("""
    **Track 1: Bioinformatics and DNA Analysis**  
    *This interactive suite directly honors the scientific legacy of **Dr. Pushpa Mittra Bhargava** (1928–2017), 
    the pioneering visionary who founded India's Centre for Cellular and Molecular Biology (CCMB). Replicating his commitment 
    to rigorous empirical research, this workspace evaluates CRISPR-Cas9 exon-cleavage validation matrices 
    against background oncology sequencing noise using real-time automated hypothesis testing modeling.*
""")
st.markdown("---")

# 2. CONFIGURATION SIDEBAR (BLOCK 1)
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

# 3. BLOCK 2: SEQUENCING CONTEXT & LIVE BIOPYTHON IO PIPELINE
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

# 4. BLOCK 3: CLINICAL TELEMETRY DASHBOARD & HEATMAP
st.subheader("📊 Step 2: Cohort Variant Deletion Matrix & Peak Telemetry")

# Initialize seed for consistent, deterministic visual simulation capturing
np.random.seed(42)
mutation_matrix = np.zeros((n_patients, n_bases))
crispr_success_mask = np.random.rand(n_patients) < crispr_efficiency

# Variable safeguards to prevent rendering text collisions
binary_options = [1, 0]

# Build variant profiling states across the sample rows
for patient_idx in range(n_patients):
    for pos in range(n_bases):
        if crispr_success_mask[patient_idx] and (target_start <= pos < target_end):
            mutation_matrix[patient_idx, pos] = np.random.choice(binary_options, p=[0.85, 0.15])
        else:
            mutation_matrix[patient_idx, pos] = np.random.choice(binary_options, p=[base_error_rate, 1 - base_error_rate])

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

# Build the custom Matplotlib rendering canvas using high-visibility 4K cinematic overrides
fig, ax = plt.subplots(figsize=(14, 5.5))
fig.patch.set_facecolor('#0b132b')  # Dark navy canvas match
ax.set_facecolor('#0b132b')

# Dark Navy Canvas Background map populated with Midnight Blue (#1c2541) and Magenta (#e056fd) node indicators
custom_cmap = sns.color_palette(["#1c2541", "#e056fd"], as_cmap=True)
sns.heatmap(df_matrix, cmap=custom_cmap, cbar=False, linewidths=1.2, linecolor="#0b132b", ax=ax)

ax.set_title("Variant Cleavage Matrix Map (Vibrant Magenta = Target Base Knocked Out / Absent)", color='#48dbfb', fontsize=12, weight='bold', pad=15, family='monospace')
ax.set_xlabel("Genomic Sequence Base Identifiers & Coordinate Index positions", color='#e0e1dd', fontsize=10, labelpad=8, family='monospace')
ax.set_ylabel("Simulated Patient Sample Fields", color='#e0e1dd', fontsize=10, family='monospace')
ax.tick_params(colors='#e0e1dd', labelsize=9)

# Drop sharp neon yellow coordinate boundary wirelines over target coordinates
ax.axvline(x=target_start, color='#fffa65', linestyle=':', linewidth=2.5, alpha=0.8)
ax.axvline(x=target_end, color='#fffa65', linestyle=':', linewidth=2.5, alpha=0.8)

# Render completed visual element to interface window
st.pyplot(fig)
plt.close(fig)  # Memory safety cleanup
st.markdown("---")

# 5. INTERACTIVE COORDINATE MUTATION OVERLAY MODULE
st.subheader("🔍 Step 3: Interactive Coordinate Mutation Overlay")
st.markdown("*Select a coordinate node to isolate specific structural variants parsed by the deep learning framework.*")

o_col1, o_col2 = st.columns(2)
with o_col1:
    selected_patient = st.selectbox("🎯 Target Patient Record Row", options=patient_ids)
with o_col2:
    selected_base = st.selectbox("🧬 Target Base Coordinate Column", options=position_labels)

node_val = df_matrix.loc[selected_patient, selected_base]
base_idx = position_labels.index(selected_base)
is_in_target = target_start <= base_idx < target_end

st.markdown(f"### 📊 Micro-Telemetry Node Audit: `{selected_patient}` ── `{selected_base}`")

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Node Structural State:**")
        if node_val == 1.0:
            st.markdown("<h3 style='color:#e056fd; margin:0;'>🧬 KNOCKOUT VARIANT PRESENT</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#1c2541; margin:0;'>⚫ PASSIVE WILDTYPE BASE</h3>", unsafe_allow_html=True)
    with c2:
        st.markdown("**CRISPR Envelope Proximity:**")
        if is_in_target:
            st.markdown("<h3 style='color:#fffa65; margin:0;'>🎯 INSIDE TARGET EXON</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#48dbfb; margin:0;'>🌐 BACKGROUND REGION</h3>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Architectural Class Assignment:**")
        if is_in_target and node_val == 1.0:
            st.info("Intended Cleavage Signalling")
        elif not is_in_target and node_val == 1.0:
            st.warning("Off-Target Sequencing Noise Event")
        else:
            st.success("Stable Unmodified Base Domain")

st.markdown("---")

# 6. COMPREHENSIVE SPREADSHEET EXPORT BLOCK
st.subheader("💾 Step 4: Streamlit Data Infrastructure Export")
st.markdown("Extract the evaluated model array below into standard formats for external pipeline ingestion steps.") 
csv_data = df_matrix.to_csv().encode('utf-8') 
st.download_button( 
    label="📥 Download Variant Cleavage Matrix Spreadsheet (.csv)", 
    data=csv_data,
    file_name="bhargava_crispr_simulation_output.csv",
    mime="text/csv"
)
