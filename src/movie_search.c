/** @file
 * @brief Scalar vector normalization and result ranking.
 */
#include "movie_search.h"
#include <math.h>
#include <stdlib.h>

void normalize_movie(const float* raw, float* normalized, uint32_t dim) {
    /* Scaling before squaring avoids overflow for large finite components. */
    float scale = 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float value = fabsf(raw[i]);
        if (value > scale)
            scale = value;
    }
    for (uint32_t i = 0; i < dim; i++) {
        float value = raw[i] / scale;
        sum += value * value;
    }
    /* Dividing by the scaled length gives the same unit vector as the raw length. */
    float norm = sqrtf(sum);
    for (uint32_t i = 0; i < dim; i++) {
        normalized[i] = (raw[i] / scale) / norm;
    }
}

/**
 * Compares results by descending score, then ascending ID for exact ties.
 * @param a Pointer to the first MovieResult.
 * @param b Pointer to the second MovieResult.
 * @return A negative, zero, or positive value according to their rank.
 */
static int compare_results(const void* a, const void* b) {
    const MovieResult* x = (const MovieResult*)a;
    const MovieResult* y = (const MovieResult*)b;
    if (x->score > y->score)
        return -1;
    if (x->score < y->score)
        return 1;
    /* Exact ties use IDs so their order does not depend on qsort's stability. */
    if (x->id < y->id)
        return -1;
    if (x->id > y->id)
        return 1;
    return 0;
}

void sort_movie_results(MovieResult* results, size_t count) {
    qsort(results, count, sizeof(*results), compare_results);
}
