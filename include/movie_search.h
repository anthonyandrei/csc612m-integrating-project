#ifndef MOVIE_SEARCH_H
#define MOVIE_SEARCH_H
#include "movie_contract.h"

/** One candidate score and its location in the loaded corpus. */
typedef struct {
    uint64_t id;      /**< Wikipedia movie ID, used to break exact score ties. */
    uint32_t index;   /**< Index into the corresponding MovieCorpus arrays. */
    float score;      /**< Scalar dot product of normalized vectors. */
} MovieResult;

/**
 * Normalizes a movie vector to unit length using float32 arithmetic.
 *
 * The input must contain dim finite values with nonzero magnitude, as ensured
 * by load_movie_corpus. Both buffers must hold dim floats; they may be the same
 * buffer. Partially overlapping buffers are not supported. dim must be positive.
 *
 * @param raw         Original vector; unchanged unless used as the output.
 * @param normalized  Buffer receiving the normalized vector.
 * @param dim         Number of vector components.
 */
void normalize_movie(const float* raw, float* normalized, uint32_t dim);

/**
 * Sorts candidate results in place by descending score and ascending movie ID
 * for exactly equal scores. Scores must be finite.
 *
 * @param results  Array of at least count results.
 * @param count    Number of results to sort.
 */
void sort_movie_results(MovieResult* results, size_t count);
#endif
