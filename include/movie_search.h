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

/**
 * Computes the dot product of two float32 vectors using 256-bit AVX2 registers.
 * Handles dimensions smaller than 8 and arbitrary leftover components via scalar tail loop.
 *
 * @param a    Pointer to first float array (must hold at least dim elements).
 * @param b    Pointer to second float array (must hold at least dim elements).
 * @param dim  Number of float elements.
 * @return     Dot product as a single-precision float in XMM0.
 */
float dot_product_simd_ymm(const float* a, const float* b, uint32_t dim);


/** Half-open range [begin, end) into a candidate array. Empty if begin == end. */
typedef struct {
    size_t begin;
    size_t end;
} CandidateRange;

#define TOPK_ERROR ((size_t)-1)

/** Returns nonzero if a ranks strictly ahead of b (score desc, then ID asc). */
int movie_result_ranks_before(const MovieResult* a, const MovieResult* b);

/**
 * Splits count candidates into parts contiguous ranges whose sizes differ by
 * at most one. When parts > count, trailing ranges are empty.
 * @return Number of ranges written (parts), or 0 if parts == 0.
 */
size_t make_even_partitions(size_t count, size_t parts, CandidateRange* out);

/** Nonzero if the ranges start at 0, are contiguous, and end exactly at count. */
int partitions_are_valid(size_t count, const CandidateRange* parts, size_t num_parts);

/**
 * Serial partitioned top-K. Each partition keeps a local K-bounded heap, then
 * the retained results are merged and sorted.
 * @param scratch  Room for num_parts * k results.
 * @param out      Room for k results; receives the final ordered matches.
 * @return Number of results written (min(k, count)), or TOPK_ERROR if k == 0
 *         or the partitions are invalid.
 */
size_t select_top_k_partitioned(const MovieResult* candidates, size_t count,
                                const CandidateRange* parts, size_t num_parts,
                                size_t k, MovieResult* scratch, MovieResult* out);

/**
 * Full-sort reference: copies candidates into work (room for count), sorts,
 * and writes the first min(k, count) results to out.
 */
size_t select_top_k_reference(const MovieResult* candidates, size_t count,
                              size_t k, MovieResult* work, MovieResult* out);

#endif