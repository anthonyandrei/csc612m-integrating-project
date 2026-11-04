CC ?= clang
CFLAGS ?= -Wall -Wextra -pedantic -std=c11 -Iinclude -O2
LDFLAGS ?= -lm

# Cross-platform Python detection (Windows vs Unix/macOS)
ifeq ($(OS),Windows_NT)
    PYTHON ?= .venv/Scripts/python.exe
    ifeq ($(wildcard $(PYTHON)),)
        PYTHON = python
    endif
else
    PYTHON ?= .venv/bin/python
    ifeq ($(wildcard $(PYTHON)),)
        PYTHON = python3
    endif
endif

HEADERS = include/movie_contract.h include/movie_loader.h
SRC = src/movie_loader.c
TEST_SRC = tests/test_loader.c
TEST_BIN = tests/test_loader

VERIFY_SRC = src/verify_corpus.c
VERIFY_BIN = verify_corpus

.PHONY: all sample-data test-loader profile-plots \
        verify_corpus prepare-full verify-full full-demo clean help

all: test-loader

# Step 1 & 2: Profile plot lengths across full CMU corpus
profile-plots:
	$(PYTHON) scripts/profile_plot_lengths.py

# Step 4: Generate small reproducible sample binary corpus (MVE1 contract)
sample-data:
	$(PYTHON) scripts/prepare_sample_vectors.py

# Step 5: Compile and run C loader validation test suite
test-loader: $(TEST_BIN)
	./$(TEST_BIN)

# Rebuild when source OR headers change (addresses review finding #4)
$(TEST_BIN): $(SRC) $(TEST_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) $(TEST_SRC) -o $(TEST_BIN) $(LDFLAGS)

# Full-Corpus Additions

# Compile the C cross-validator
$(VERIFY_BIN): $(VERIFY_SRC) $(SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(VERIFY_SRC) $(SRC) -o $(VERIFY_BIN) $(LDFLAGS)

verify_corpus: $(VERIFY_BIN)

# Generate the full binary, manifest, and rejection report
prepare-full:
	$(PYTHON) scripts/prepare_full_vectors.py

# Validate the binary against the manifest count using C
verify-full: verify_corpus
	@EXPECTED=$$($(PYTHON) -c "import json;print(json.load(open('data/reports/manifest.json'))['output']['num_movies'])") ; \
	./$(VERIFY_BIN) data/full_movies.bin $$EXPECTED

full-demo: prepare-full verify-full
	@echo "========================================================="
	@echo " SUCCESS: Full corpus generated and C-verified."
	@echo "========================================================="

# End of Full-Corpus Additions

clean:
	rm -f $(TEST_BIN) $(VERIFY_BIN) *.o src/*.o tests/*.o data/temp_test_*.bin

help:
	@echo "Available make targets:"
	@echo "  make test-loader    - Compile and run the C loader validation suite"
	@echo "  make sample-data    - Generate data/sample_movies.bin via Python offline script"
	@echo "  make profile-plots  - Profile CMU plot lengths against Sentence-BERT limit"
	@echo "  make verify_corpus  - Build the C cross-validator for full_movies.bin"
	@echo "  make prepare-full   - Run full-corpus preparation pipeline"
	@echo "  make verify-full    - Validate full_movies.bin against manifest"
	@echo "  make full-demo      - prepare-full + verify-full (Week 10 demo)"
	@echo "  make clean          - Remove compiled binaries and test artifacts"