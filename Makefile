CC ?= clang
CFLAGS ?= -Wall -Wextra -pedantic -std=c11 -Iinclude -O2
LDFLAGS ?= -lm
PYTHON ?= .venv/bin/python

SRC = src/movie_loader.c
TEST_SRC = tests/test_loader.c
TEST_BIN = tests/test_loader

.PHONY: all sample-data test-loader profile-plots clean help

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

$(TEST_BIN): $(SRC) $(TEST_SRC)
	$(CC) $(CFLAGS) $(SRC) $(TEST_SRC) -o $(TEST_BIN) $(LDFLAGS)

clean:
	rm -f $(TEST_BIN) *.o src/*.o tests/*.o data/temp_test_*.bin

help:
	@echo "Available make targets:"
	@echo "  make test-loader    - Compile and run the C loader validation suite"
	@echo "  make sample-data    - Generate data/sample_movies.bin via Python offline script"
	@echo "  make profile-plots  - Profile CMU plot lengths against Sentence-BERT limit"
	@echo "  make clean          - Remove compiled binaries and test artifacts"
