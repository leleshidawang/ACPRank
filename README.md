# ACPRank
ACPRank, a computational framework based on cost-aware ranking learning designed to discover potent anticancer peptides

## Stage 1: Protease Analysis and Model Building (01_analyze_protease_batch.py)

### 📊 Overview

`01_analyze_protease_batch.py` is the first stage of the ACPRank pipeline. It analyzes experimentally validated protease cleavage sites from the MEROPS database and constructs position-specific scoring models for each protease. These models learn the amino acid preferences at different positions around cleavage sites.

### 🔄 Workflow

```
┌─────────────────────────────────────────────────────┐
│  Input: MEROPS Database + Protein Sequences         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  1. Query Experimental Cleavage Data                │
│     - Fetch validated substrate proteins            │
│     - Extract cleavage positions (P1-P1' bonds)     │
│     - Download substrate sequences                  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  2. Extract Position-Specific Amino Acids           │
│     - Extract context windows (P5-P5')              │
│     - Analyze amino acid frequencies                │
│     - Calculate position-specific properties        │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  3. Build Frequency Matrix                          │
│     - Calculate amino acid frequency at each pos    │
│     - Compute information entropy                   │
│     - Derive position weights (0-1)                 │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  4. Determine Prediction Threshold                  │
│     - Score known cleavage sites                    │
│     - Use minimum true-positive score as threshold  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  Output: Prediction Model + Detailed Reports        │
└─────────────────────────────────────────────────────┘
```

### 📥 Input Files

#### 1. **merops_valid_id_mapping.csv** (Required)
Configuration file mapping gene names to MEROPS database identifiers.

**Format:**
```csv
gene_name,uniprot_id,merops_id
CTSB,P07711,MA0001
CTSL,P07711,MA0002
CTSK,P07711,MA0003
```

**Columns:**
- `gene_name`: Human protease gene name
- `uniprot_id`: UniProt protein identifier
- `merops_id`: MEROPS protease classification ID

#### 2. **MEROPS Database** (Required)
MySQL database containing experimentally verified protease cleavage data.

**Required Table Schema:**
```sql
CREATE TABLE cleavage (
    code VARCHAR(20),              -- MEROPS protease ID
    uniprot_acc VARCHAR(10),       -- UniProt accession of substrate
    p1 INT,                        -- Position of P1 amino acid
    cleavage_type VARCHAR(50),     -- Type of cleavage
    cleavage_evidence VARCHAR(50)  -- Evidence type: 'experimental' or other
);
```

**Example Data:**
```
code     | uniprot_acc | p1   | cleavage_type | cleavage_evidence
---------|-------------|------|---------------|------------------
MA0001   | P07711      | 120  | single        | experimental
MA0001   | P07711      | 456  | single        | experimental
MA0002   | Q16378      | 235  | single        | experimental
```

### 📤 Output Files

The script generates a hierarchical output structure in the `protease_analysis_batch/` directory:

```
protease_analysis_batch/
├── prediction_models.pkl                          [Binary Model Cache]
│   └── Contains all trained models for 02_*.py
│
├── protease_site_statistics.csv                   [Summary Statistics]
│   └── One row per protease with model status
│
├── {uniprot_id}/                                  [Per-Protease Results]
│   ├── {merops_id}_substrate_analysis.txt        [Detailed Report]
│   │   ├── Cleavage site locations with contexts
│   │   ├── Amino acid frequency tables (P5-P5')
│   │   ├── Position entropy & weight analysis
│   │   └── Prediction score statistics
│   │
│   └── {merops_id}_peptides.csv                  [Predicted Peptides]
│       └── Intermediate peptides from predicted sites
│
├── missing_sequences_peptides.csv                [Optional Output]
└── processed_by_SignalP6_peptides.csv            [Optional Output]
```

#### Key Output File Details

**1. prediction_models.pkl** (Binary format)
```python
{
  'MA0001': {
    'frequency_matrix': {
      'A': {'P5': 0.15, 'P4': 0.20, 'P3': 0.18, 'P2': 0.22, 
            'P1': 0.25, 'P1\'': 0.20, 'P2\'': 0.18, ...},
      'C': {...},
      ...  # 20 amino acids
    },
    'position_weights': {
      'P5': 0.08, 'P4': 0.10, 'P3': 0.12, 'P2': 0.15,
      'P1': 0.20, 'P1\'': 0.18, 'P2\'': 0.10, ...
    },
    'min_tp_score': 0.45  # Prediction threshold
  },
  'MA0002': {...},
  ...
}
```

**2. protease_site_statistics.csv**
```csv
gene_name,uniprot_id,merops_id,experimental_sites,unique_substrates,has_model,status
CTSB,P07711,MA0001,156,45,True,Success
CTSL,Q16378,MA0002,0,0,False,No experimental data
CTSK,P07711,MA0003,89,32,True,Success
```

**3. {merops_id}_substrate_analysis.txt** (Example section)
```
# Comprehensive Cleavage Site Analysis for Protease MA0001

## Amino Acid Frequency at Each Position

| Amino Acid |   P5   |   P4   |   P3   |   P2   |   P1   |   P1'  |   P2'  |   P3'  |   P4'  |   P5'  |
|------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **A**     | 15.30% | 12.40% | 13.80% | 14.20% | 16.50% | 18.20% | 15.60% | 12.40% |  9.50% |  7.30% |
| **C**     |  2.10% |  1.80% |  2.30% |  1.90% |  2.50% |  3.10% |  2.80% |  1.60% |  1.20% |  0.80% |
...

## Position Entropy and Weights

| Position | Entropy | Normalized Weight |
|----------|---------|-------------------|
| P5       | 3.8452  |     0.0843        |
| P4       | 3.9104  |     0.0987        |
| P3       | 3.8901  |     0.1124        |
| P2       | 3.7832  |     0.1456        |
| P1       | 3.6543  |     0.1998        |
| P1'      | 3.7125  |     0.1876        |
| P2'      | 3.8234  |     0.1243        |
...

### Determined Threshold Score

The minimum score among true positive predictions is: **0.456789**
This value will be used as the threshold for predictions on unknown sequences.
```

### ⚙️ Key Functions

#### 1. `get_protease_substrates(code)`
**Purpose:** Retrieves all known cleavage sites for a specific protease from MEROPS database.

**Input:** MEROPS protease ID (e.g., "MA0001")

**Output:** Dictionary mapping UniProt IDs to list of cleavage sites
```python
{
  'P07711': [
    {'code': 'MA0001', 'uniprot_acc': 'P07711', 'p1': 120, ...},
    {'code': 'MA0001', 'uniprot_acc': 'P07711', 'p1': 456, ...},
  ],
  'Q16378': [...]
}
```

#### 2. `extract_position_specific_aas(exp_sites, window=5)`
**Purpose:** Extracts amino acid context (P5-P5') around each cleavage site.

**Key Variables:**
- **P-positions:** Residues N-terminal to cleavage (P5, P4, P3, P2, P1)
- **P'-positions:** Residues C-terminal to cleavage (P1', P2', P3', P4', P5')
- The bond cleaved is between P1 and P1'

**Output:**
```python
{
  'P5': ['L', 'M', 'F', 'Y', ...],    # List of P5 residues
  'P4': ['V', 'A', 'S', 'T', ...],
  'P3': ['P', 'Q', 'R', 'K', ...],
  'P2': ['S', 'T', 'E', 'D', ...],
  'P1': ['L', 'V', 'A', 'Y', ...],    # Position of cleavage
  'P1\'': ['V', 'E', 'L', 'T', ...],  # N-terminal of new C-terminus
  ...
}
```

#### 3. `build_frequency_matrix(position_properties)`
**Purpose:** Calculates amino acid frequency at each position.

**Calculation:**
```
For each position (P5-P5'):
  For each amino acid (20 standard):
    frequency[aa][position] = count(aa at position) / total_observations
```

**Output Matrix Structure:**
```python
{
  'A': {'P5': 0.153, 'P4': 0.124, 'P3': 0.138, ...},
  'C': {'P5': 0.021, 'P4': 0.018, 'P3': 0.023, ...},
  'D': {'P5': 0.082, 'P4': 0.095, 'P3': 0.078, ...},
  ...
}
```

#### 4. `calculate_position_entropy(frequency_matrix, positions)`
**Purpose:** Computes Shannon entropy and derives position weights.

**Output:**
```python
entropy_values = {'P5': 3.845, 'P4': 3.910, 'P3': 3.890, 'P2': 3.783, 'P1': 3.654, ...}
position_weights = {'P5': 0.084, 'P4': 0.099, 'P3': 0.112, 'P2': 0.146, 'P1': 0.200, ...}
```

#### 5. `score_cleavage_site(sequence, position, frequency_matrix, position_weights)`
**Purpose:** Scores a potential cleavage site using the learned model.

**Algorithm:**
```
For each position around the cleavage site:
  1. Extract amino acid at that position
  2. Look up frequency in matrix
  3. Multiply by position weight
  4. Accumulate weighted sum
5. Normalize by total weights used
```

**Example Scoring:**
```
Sequence: ...LVAQSA↓VEL...
Position 7 (P1-P1' bond marked by ↓)

P5(L): freq=0.15, weight=0.084 → contribution = 0.0126
P4(V): freq=0.18, weight=0.099 → contribution = 0.0178
P3(A): freq=0.22, weight=0.112 → contribution = 0.0246
P2(Q): freq=0.05, weight=0.146 → contribution = 0.0073
P1(S): freq=0.12, weight=0.200 → contribution = 0.0240
P1'(A): freq=0.20, weight=0.188 → contribution = 0.0376
P2'(V): freq=0.16, weight=0.124 → contribution = 0.0198
P3'(E): freq=0.11, weight=0.105 → contribution = 0.0116
P4'(L): freq=0.17, weight=0.052 → contribution = 0.0088
P5'(X): No data

Final Score = (0.0126 + 0.0178 + 0.0246 + 0.0073 + 0.0240 + 0.0376 + 0.0198 + 0.0116 + 0.0088) / total_weights
           = 0.1641 / (0.084+0.099+0.112+0.146+0.200+0.188+0.124+0.105+0.052)
           = 0.1641 / 1.110 ≈ 0.148
```

### 🎯 Usage

#### Basic Usage
```bash
python3 01_analyze_protease_batch.py
```

#### Analyze Only (skip prediction phase)
```bash
python3 01_analyze_protease_batch.py --analyze-only
```

#### Prediction Only (reuse existing models)
```bash
python3 01_analyze_protease_batch.py --predict-only
```

### 🔧 Configuration

Edit database connection in the script:
```python
db_config = {
    'user': 'root',
    'database': 'merops',
    'unix_socket': '/var/run/mysqld/mysqld.sock'
}
```

