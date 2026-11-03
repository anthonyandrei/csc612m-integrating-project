ifeq ($(origin CC),default)
    CC = clang
endif
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

HEADERS = include/movie_contract.h include/movie_loader.h include/movie_search.h
SRC = src/movie_loader.c
TEST_SRC = tests/test_loader.c
TEST_BIN = tests/test_loader
SEARCH_BIN = tests/movie_search
# Keep the benchmark scalar even when optimization is enabled.
SCALAR_FLAGS ?= -fno-vectorize -fno-slp-vectorize -ffp-contract=off

.PHONY: all sample-data test test-loader profile-plots clean help

all: $(SEARCH_BIN)

# Step 1 & 2: Profile plot lengths across full CMU corpus
profile-plots:
	$(PYTHON) scripts/profile_plot_lengths.py

# Step 4: Generate small reproducible sample binary corpus (MVE1 contract)
sample-data:
	$(PYTHON) scripts/prepare_sample_vectors.py

# Step 5: Compile and run C loader validation test suite
test-loader: $(TEST_BIN)
	./$(TEST_BIN)

test: test-loader

$(SEARCH_BIN): $(SRC) src/movie_search.c src/main.c $(HEADERS)
	$(CC) $(CFLAGS) $(SCALAR_FLAGS) $(SRC) src/movie_search.c src/main.c -o $(SEARCH_BIN) $(LDFLAGS)

# Rebuild when source OR headers change (addresses review finding #4)
$(TEST_BIN): $(SRC) $(TEST_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SRC) $(TEST_SRC) -o $(TEST_BIN) $(LDFLAGS)

clean:
	rm -f $(TEST_BIN) $(SEARCH_BIN) $(TEST_BIN).exe $(SEARCH_BIN).exe *.o src/*.o tests/*.o data/temp_test_*.bin

help:
	@echo "Available make targets:"
	@echo "  make test           - Run loader checks"
	@echo "  make test-loader    - Compile and run the C loader validation suite"
	@echo "  make sample-data    - Generate data/sample_movies.bin via Python offline script"
	@echo "  make profile-plots  - Profile CMU plot lengths against Sentence-BERT limit"
	@echo "  make clean          - Remove compiled binaries and test artifacts"
