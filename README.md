# SIMD and Multicore Acceleration for Movie Recommendations

Integrating project for **CSC612M (Advanced Computer Architecture)**, 1st Term SY 2026-2027, De La Salle University.

## Proposal
- **Document**: [Proposal Document](docs/proposal-doc.pdf)
- **Video**: [Proposal Pitch Video](https://drive.google.com/file/d/1ja09MC404rA_IRstCRpHqdqC-wjBU5wy/view?usp=sharing)

---

## Movie-Vector Pipeline & Setup Instructions

The repository uses an offline preparation pipeline (Python) to generate compact binary datasets (`MVE1` format) that are directly loaded by the runtime C and AVX2 SIMD search programs without any runtime ML dependency.

### 1. Environment Setup

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

#### Windows (Command Prompt / PowerShell)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r scripts\requirements.txt
```

---

### 2. Dataset Download & Extraction

The project uses the **CMU Movie Summary Corpus** (Bamman et al., ACL 2013).

#### Linux / macOS
```bash
mkdir -p data/raw
curl -L -o data/raw/MovieSummaries.tar.gz http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
tar -xzf data/raw/MovieSummaries.tar.gz -C data/raw/ MovieSummaries/plot_summaries.txt MovieSummaries/movie.metadata.tsv
```

#### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path data\raw
Invoke-WebRequest -Uri "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz" -OutFile "data\raw\MovieSummaries.tar.gz"
tar -xzf data\raw\MovieSummaries.tar.gz -C data\raw\ MovieSummaries/plot_summaries.txt MovieSummaries/movie.metadata.tsv
```

---

### 3. Running Preparation & Verification

#### Profile CMU Corpus Plot Lengths
Measures the 42,306 plot summaries against the 256-token limit of `all-MiniLM-L6-v2`:
```bash
make profile-plots
# Or directly via Python:
python scripts/profile_plot_lengths.py
```

#### Generate Sample Movie Vectors
Generates `data/sample_movies.bin` (16 real movies encoded via Sentence-BERT with chunking + mean-pooling):
```bash
make sample-data
# Or directly via Python:
python scripts/prepare_sample_vectors.py
```

#### Run C Loader Test Suite
Compiles and executes the validation test suite across all 9 test cases (verifying happy path, 32-byte AVX2 alignment, coordinate matching, and 7 error rejection modes):
```bash
make test-loader
```
On Windows with MSVC:
```cmd
cl /W4 /O2 /Iinclude src\movie_loader.c tests\test_loader.c /Fe:tests\test_loader.exe
tests\test_loader.exe
```
