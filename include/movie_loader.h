#ifndef MOVIE_LOADER_H
#define MOVIE_LOADER_H

#include "movie_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Loads and validates a binary movie corpus file adhering to the MVE1 specification.
 *
 * Validation checks performed:
 * 1. File accessibility.
 * 2. Magic identifier validation (must be 'MVE1').
 * 3. Empty corpus detection (num_movies > 0).
 * 4. Dimension match verification (if expected_dim > 0).
 * 5. Strict file size verification against computed byte size.
 * 6. Vector value validation: rejects non-finite floats (NaN, +Inf, -Inf).
 * 7. Zero-magnitude vector validation: rejects vectors with L2 norm < 1e-6.
 *
 * @param filepath      Path to the .bin file.
 * @param expected_dim  Expected vector dimension (e.g., 384), or 0 to accept any.
 * @param out_corpus    Pointer receiving the allocated MovieCorpus structure.
 * @return CorpusStatus error code (CORPUS_OK on success).
 */
CorpusStatus load_movie_corpus(const char* filepath, uint32_t expected_dim, MovieCorpus** out_corpus);

/**
 * Frees all memory allocated for a MovieCorpus structure.
 *
 * @param corpus Pointer to the MovieCorpus to deallocate.
 */
void free_movie_corpus(MovieCorpus* corpus);

#ifdef __cplusplus
}
#endif

#endif /* MOVIE_LOADER_H */
