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

## Scalar search baseline

The search accepts an MVE1 file, a Wikipedia movie ID, and K. K must be an integer from 1 through the number of movies minus one. The query must exist. The loader rejects the whole file for invalid vectors, zero dimensions, zero or duplicate IDs, or empty titles. Results exclude the query and are sorted by descending float32 score; exact ties use ascending movie ID. Result lines are tab-separated; titles escape backslash, tab, carriage return, and newline as `\\`, `\t`, `\r`, and `\n`.

On Windows x64 with a Visual Studio Developer Command Prompt (MSVC), build the release scalar baseline and checks:

```cmd
cl /std:c11 /W4 /O2 /Qvec- /fp:precise /Iinclude src\movie_loader.c src\movie_search.c src\main.c /Fe:tests\movie_search.exe
cl /std:c11 /W4 /O2 /Iinclude src\movie_loader.c tests\test_loader.c /Fe:tests\test_loader.exe
tests\test_loader.exe
tests\movie_search.exe data\sample_movies.bin 30007 5
```

Record `cl` output to identify the compiler version for each benchmark. On the development machine, the alternate tested compiler was clang 17.0.6 targeting `x86_64-w64-windows-gnu` with `-O2 -fno-vectorize -fno-slp-vectorize -ffp-contract=off`. With clang and make installed, `make CC=clang` builds the program and `make CC=clang test` runs the loader checks. The sample is already included; it does not require Python packages or encoder inference to run.

The executable prints the movie count, dimension, query ID, K, one warmup, 30 measured runs, candidate normalization time, and median query time in seconds. Candidate normalization is a one-time measurement. Each query includes normalization of the original query vector, scoring every other movie, and full sorting; loading and printing are outside the query timer. Record the CPU, operating system, compiler version, build command, and input file with any reported timings. Keep the input and query fixed across repetitions.

The public `sort_movie_results` function preserves full-sort ranking for later comparisons using the same scores. Automated search correctness checks and an independent double-precision reference are not currently included.
