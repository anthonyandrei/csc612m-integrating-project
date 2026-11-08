/** @file
 * @brief Scalar vector normalization and result ranking.
 */
#include "movie_search.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

int movie_result_ranks_before(const MovieResult* a, const MovieResult* b) {
    return compare_results(a, b) < 0;
}

size_t make_even_partitions(size_t count, size_t parts, CandidateRange* out) {
    if (parts == 0)
        return 0;
    size_t base = count / parts, extra = count % parts, begin = 0;
    for (size_t p = 0; p < parts; p++) {
        size_t size = base + (p < extra ? 1 : 0);
        out[p].begin = begin;
        out[p].end = begin + size;
        begin += size;
    }
    return parts;
}

int partitions_are_valid(size_t count, const CandidateRange* parts, size_t num_parts) {
    size_t expected_begin = 0;
    if (num_parts == 0)
        return 0;
    for (size_t p = 0; p < num_parts; p++) {
        if (parts[p].begin != expected_begin || parts[p].end < parts[p].begin)
            return 0;
        expected_begin = parts[p].end;
    }
    return expected_begin == count;
}

/* Heap root is the worst-ranked retained result, so it is the one to evict. */
static void heap_swap(MovieResult* a, MovieResult* b) {
    MovieResult t = *a;
    *a = *b;
    *b = t;
}

static void heap_sift_up(MovieResult* heap, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!movie_result_ranks_before(&heap[parent], &heap[i]))
            return;
        heap_swap(&heap[parent], &heap[i]);
        i = parent;
    }
}

static void heap_sift_down(MovieResult* heap, size_t size, size_t i) {
    for (;;) {
        size_t worst = i, left = 2 * i + 1, right = left + 1;
        if (left < size && movie_result_ranks_before(&heap[worst], &heap[left]))
            worst = left;
        if (right < size && movie_result_ranks_before(&heap[worst], &heap[right]))
            worst = right;
        if (worst == i)
            return;
        heap_swap(&heap[worst], &heap[i]);
        i = worst;
    }
}

/* Retains the best min(k, n) of c[0..n) in heap; returns the retained count. */
static size_t heap_select(const MovieResult* c, size_t n, size_t k, MovieResult* heap) {
    size_t size = 0;
    for (size_t i = 0; i < n; i++) {
        if (size < k) {
            heap[size] = c[i];
            heap_sift_up(heap, size);
            size++;
        } else if (movie_result_ranks_before(&c[i], &heap[0])) {
            heap[0] = c[i];
            heap_sift_down(heap, size, 0);
        }
    }
    return size;
}

size_t select_top_k_partitioned(const MovieResult* candidates, size_t count,
                                const CandidateRange* parts, size_t num_parts,
                                size_t k, MovieResult* scratch, MovieResult* out) {
    if (k == 0 || !partitions_are_valid(count, parts, num_parts))
        return TOPK_ERROR;
    size_t retained = 0;
    for (size_t p = 0; p < num_parts; p++) {
        retained += heap_select(candidates + parts[p].begin,
                                parts[p].end - parts[p].begin, k, scratch + retained);
    }
    sort_movie_results(scratch, retained);
    size_t n = retained < k ? retained : k;
    memcpy(out, scratch, n * sizeof(*out));
    return n;
}

size_t select_top_k_reference(const MovieResult* candidates, size_t count,
                              size_t k, MovieResult* work, MovieResult* out) {
    memcpy(work, candidates, count * sizeof(*work));
    sort_movie_results(work, count);
    size_t n = count < k ? count : k;
    memcpy(out, work, n * sizeof(*out));
    return n;
}

#if !defined(__x86_64__) && !defined(_M_X64)
/**
 * Emulated fallback for non-x86_64 platforms (e.g. ARM64 development hosts).
 * On x86-64 Windows / Linux, this is overridden by src/dot_simd_x64.asm.
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
float dot_product_simd_ymm(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif