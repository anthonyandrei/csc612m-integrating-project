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

The scalar search benchmark targets Windows; its timer uses Windows performance counters.

The search accepts an MVE1 file, a Wikipedia movie ID, and K. K must be an integer from 1 through the number of movies minus one. The query must exist. The loader rejects the whole file for invalid vectors, zero dimensions, zero or duplicate IDs, or empty titles. Results exclude the query and are sorted by descending float32 score; exact ties use ascending movie ID. Result lines are tab-separated; titles escape backslash, tab, carriage return, and newline as `\\`, `\t`, `\r`, and `\n`.

On Windows x64 with a Visual Studio Developer Command Prompt (MSVC), build the release scalar baseline and checks:

```cmd
cl /std:c11 /W4 /O2 /Qvec- /fp:precise /Iinclude src\movie_loader.c src\movie_search.c src\main.c /Fe:tests\movie_search.exe
cl /std:c11 /W4 /O2 /Iinclude src\movie_loader.c tests\test_loader.c /Fe:tests\test_loader.exe
cl /std:c11 /W4 /O2 /Iinclude tests\test_search.c /Fe:tests\test_search.exe
tests\test_loader.exe
tests\test_search.exe
tests\movie_search.exe data\sample_movies.bin 30007 5
```

Record `cl` output to identify the compiler version for each benchmark. On the development machine, the alternate tested compiler was clang 17.0.6 targeting `x86_64-w64-windows-gnu` with `-O2 -fno-vectorize -fno-slp-vectorize -ffp-contract=off`. With clang and make installed, `make CC=clang` builds the program and `make CC=clang test` runs the loader and search checks. The sample is already included; it does not require Python packages or encoder inference to run.

MSVC 19.44.35229 reports that it ignores `/Qvec-`. The MSVC command has not been verified to disable automatic vectorization; use the clang scalar build for benchmark comparisons until an MSVC setting is verified.

The executable prints the movie count, dimension, query ID, K, one warmup, 30 measured runs, candidate normalization time, and median query time in milliseconds. Candidate normalization is a one-time measurement. Each query includes normalization of the original query vector, scoring every other movie, and full sorting; loading and printing are outside the query timer. Record the CPU, operating system, compiler version, build command, and input file with any reported timings. Keep the input and query fixed across repetitions.

The public `sort_movie_results` function preserves full-sort ranking for later comparisons using the same scores. The search checks generate a small MVE1 corpus, run the executable, and compare scores against cosine similarity computed independently in double precision. The absolute tolerance is `1e-5` for these three-dimensional inputs; it is not a proven error bound for arbitrary dimensions. The checks cover query exclusion, result IDs and titles, descending ranking, exact ties, invalid K and query IDs, invalid vectors and metadata, and large finite components. A failed check returns a nonzero exit status. Run them from the repository root with `make test-search` or the MSVC commands above.
---

## 4. Full-Corpus Preparation

This pipeline processes the **entire CMU corpus** using the same encoder, chunking policy, and input contract as the sample generator. It produces the full binary input and verifies it end to end.

### Provenance & Reproducibility

| Input          | Source                                          | Pinned By                         |
| -------------- | ----------------------------------------------- | --------------------------------- |
| Plot summaries | CMU Movie Summary Corpus (`plot_summaries.txt`) | SHA-256 in `manifest.json`        |
| Movie metadata | CMU Movie Summary Corpus (`movie.metadata.tsv`) | SHA-256 in `manifest.json`        |
| Encoder        | `sentence-transformers/all-MiniLM-L6-v2`        | Model revision in `manifest.json` |

No credentials are required and no local state is assumed. Every run regenerates the binary entirely from the two CMU tarball members downloaded in Step 2.

### Reproducible Commands

```bash
# Generate the full binary, rejection report, and manifest
make prepare-full

# Validate the binary against the manifest using the C loader
make verify-full

# Or run both in one step
make full-demo
```

### Expected Output & Usable Count

The pipeline does not assume all 42,306 plot summaries become usable movies. After cleaning, metadata matching, and duplicate handling, the accepted count is written to `data/reports/manifest.json` under `counts.accepted`.

Rejected records are logged in `data/reports/rejections.csv`, one per row, with one of these reasons:

| Reason                | Meaning                                                  |
| --------------------- | -------------------------------------------------------- |
| `malformed_line`      | Line does not have exactly 2 tab-separated fields        |
| `duplicate_id`        | `wiki_id` was already accepted (see Duplicate-ID Policy) |
| `no_metadata_match`   | `wiki_id` has no entry in `movie.metadata.tsv`           |
| `empty_title`         | Metadata title is blank                                  |
| `empty_plot`          | Plot summary is blank after whitespace trimming          |
| `encode_error:<Type>` | Encoder raised an exception                              |
| `wrong_dim`           | Encoder returned a vector that is not 384-dimensional    |
| `non_finite`          | Vector contains NaN or Inf                               |
| `zero_magnitude`      | L2 norm < 1e-6                                           |

### Duplicate-ID Policy

`plot_summaries.txt` may contain the same `wiki_id` more than once. The first occurrence in file order is accepted. Every later occurrence is rejected with reason `duplicate_id` and logged to the CSV. This is deterministic and reproducible.

### Output Artifacts

- `data/full_movies.bin`: the compiled MVE1 dataset (32-byte aligned, dim=384). Excluded from Git due to size; regenerate with `make prepare-full`.
- `data/reports/manifest.json`: provenance, SHA-256 hashes, encoder config, and accepted/rejected counts.
- `data/reports/rejections.csv`: every rejected movie, one per row, with the reason.

### C-Side Verification

`make verify-full` compiles `src/verify_corpus.c` and runs it against the generated binary. It:

1. Loads `data/full_movies.bin` with the same `load_movie_corpus()` used by the search program.
2. Confirms the loaded `num_movies` equals `manifest.json` → `output.num_movies`.
3. Re-validates every record: non-zero `wiki_id`, non-empty title, all 384 floats finite, L2 norm ≥ 1e-6.

A mismatch causes a non-zero exit status, which halts `make` and surfaces the failure.
