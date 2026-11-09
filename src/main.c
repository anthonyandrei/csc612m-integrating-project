/** @file
 * @brief Command-line scalar movie search and timing benchmark.
 */
#include "movie_loader.h"
#include "movie_search.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
    #include <windows.h>
#else
    #define _POSIX_C_SOURCE 200809L
    #include <time.h>
#endif

/** Number of timed searches after the warmup. */
#define RUNS 30

/** Largest partition count accepted by the partitioned mode. */
#define MAX_PARTITIONS 4096

/**
 * Portable monotonic time in milliseconds.
 * Uses QueryPerformanceCounter on Windows, clock_gettime on POSIX.
 */
static double get_time_ms(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return ((double)counter.QuadPart * 1000.0) / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((double)ts.tv_sec * 1000.0) + ((double)ts.tv_nsec / 1000000.0);
#endif
}

/**
 * Compares two durations for ascending qsort order.
 * @param a Pointer to the first double.
 * @param b Pointer to the second double.
 * @return A negative, zero, or positive value according to their order.
 */
static int compare_double(const void* a, const void* b) {
    double x = *(const double*)a;
    double y = *(const double*)b;
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

/**
 * Parses a positive decimal integer without signs or trailing text.
 * @param text Null-terminated input string.
 * @param value Receives the parsed value on success.
 * @return 1 on success, or 0 for invalid input or overflow.
 */
static int parse_positive(const char* text, uint64_t* value) {
    char* end;
    if (*text < '0' || *text > '9')
        return 0;
    errno = 0;
    unsigned long long n = strtoull(text, &end, 10);
    if (errno || *end || n == 0)
        return 0;
    *value = (uint64_t)n;
    return 1;
}

/**
 * Prints a title with backslashes and control delimiters escaped.
 * @param title Null-terminated title to print to standard output.
 */
static void print_title(const char* title) {
    for (; *title; title++) {
        if (*title == '\\')
            fputs("\\\\", stdout);
        else if (*title == '\t')
            fputs("\\t", stdout);
        else if (*title == '\n')
            fputs("\\n", stdout);
        else if (*title == '\r')
            fputs("\\r", stdout);
        else
            putchar((unsigned char)*title);
    }
}

/**
 * Scalar dot product of two float32 vectors, accumulated left to right.
 * @param a First vector of dim floats.
 * @param b Second vector of dim floats.
 * @param dim Number of components.
 * @return The dot product.
 */
static inline float scalar_dot(const float* a, const float* b, uint32_t dim) {
    float score = 0.0f;
    for (uint32_t d = 0; d < dim; d++)
        score += a[d] * b[d];
    return score;
}

/**
 * Normalizes the query, then scores every other movie against it in corpus order.
 *
 * Every search mode calls this, so query normalization, query exclusion and
 * score accumulation cannot drift apart between the scalar, SIMD and
 * partitioned paths. Only the dot product differs, chosen by use_simd.
 * @param corpus Loaded corpus.
 * @param query Corpus index of the query movie, which is skipped.
 * @param normalized_vectors Normalized candidate vectors, num_movies * vector_dim floats.
 * @param normalized_query Scratch buffer for the normalized query, vector_dim floats.
 * @param results Receives one MovieResult per candidate; room for num_movies - 1.
 * @param use_simd Nonzero to score with the AVX2 kernel, zero for the scalar loop.
 * @return Number of candidates written, num_movies - 1.
 */
static size_t score_candidates(const MovieCorpus* corpus, uint32_t query,
                               const float* normalized_vectors, float* normalized_query,
                               MovieResult* results, int use_simd) {
    uint32_t dim = corpus->vector_dim;
    normalize_movie(corpus->vectors + (size_t)query * dim, normalized_query, dim);
    size_t count = 0;
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        if (i == query)
            continue;
        const float* candidate = normalized_vectors + (size_t)i * dim;
        results[count].id = corpus->wiki_ids[i];
        results[count].index = i;
        float score;
        if (use_simd)
            score = dot_product_simd_ymm(normalized_query, candidate, dim);
        else
            score = scalar_dot(normalized_query, candidate, dim);
        results[count].score = score;
        count++;
    }
    return count;
}

enum SearchMode {
    MODE_SCALAR = 0,
    MODE_SIMD = 1,
    MODE_BOTH = 2,
    MODE_PARTITIONED = 3
};

/**
 * Runs a fixed-query benchmark and prints its ordered top-K results.
 *
 * Arguments are the corpus path, query movie ID, K, and optional mode
 * [scalar|simd|both|partitioned]. The partitioned mode accepts one more
 * optional argument, the partition count (default 4).
 * One warmup precedes RUNS measured searches; loading and printing are excluded
 * from query timing. Candidate normalization is timed separately.
 * In partitioned mode, scoring, partitioning, per-partition heap selection and
 * merging are all inside the timed query, and the result is verified against a
 * full-sort reference (untimed) before printing.
 * @param argc Argument count, including the program name; 4 to 6.
 * @param argv Argument strings: program name, corpus path, query ID, K, optional mode,
 *             and optional partition count.
 * @return 0 on success, or 1 for invalid input, an allocation failure, or a
 *         partitioned-result mismatch.
 */
int main(int argc, char** argv) {
    uint64_t id, k;
    uint64_t partition_count = 4;
    MovieCorpus* corpus = NULL;
    enum SearchMode mode = MODE_BOTH;

    if (argc == 5 || argc == 6) {
        if (strcmp(argv[4], "scalar") == 0 || strcmp(argv[4], "--scalar") == 0)
            mode = MODE_SCALAR;
        else if (strcmp(argv[4], "simd") == 0 || strcmp(argv[4], "--simd") == 0)
            mode = MODE_SIMD;
        else if (strcmp(argv[4], "both") == 0 || strcmp(argv[4], "--both") == 0)
            mode = MODE_BOTH;
        else if (strcmp(argv[4], "partitioned") == 0 || strcmp(argv[4], "--partitioned") == 0)
            mode = MODE_PARTITIONED;
        else {
            fprintf(stderr, "Unknown mode: %s (expected scalar, simd, both, or partitioned)\n", argv[4]);
            return 1;
        }
        if (argc == 6 && (mode != MODE_PARTITIONED ||
                          !parse_positive(argv[5], &partition_count) ||
                          partition_count > MAX_PARTITIONS)) {
            fprintf(stderr, "Usage: movie_search <corpus.bin> <query-id> <K> partitioned [1..%d partitions]\n",
                    MAX_PARTITIONS);
            return 1;
        }
    } else if (argc != 4) {
        fprintf(stderr, "Usage: movie_search <corpus.bin> <query-id> <K> [scalar|simd|both|partitioned [partitions]]\n");
        return 1;
    }

    if (!parse_positive(argv[2], &id) || !parse_positive(argv[3], &k)) {
        fprintf(stderr, "Usage: movie_search <corpus.bin> <query-id> <K> (positive integers)\n");
        return 1;
    }
    CorpusStatus status = load_movie_corpus(argv[1], 0, &corpus);
    if (status != CORPUS_OK) {
        fprintf(stderr, "Invalid corpus: %s\n", corpus_status_to_string(status));
        return 1;
    }
    uint32_t query = corpus->num_movies;
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        if (corpus->wiki_ids[i] == id)
            query = i;
    }
    /* The query itself cannot be a result, leaving at most N - 1 candidates. */
    if (query == corpus->num_movies || k >= corpus->num_movies) {
        fprintf(stderr, "Unknown query ID or K outside 1..N-1\n");
        free_movie_corpus(corpus);
        return 1;
    }
    uint32_t vector_dim = corpus->vector_dim;
    float* normalized_vectors = (float*)malloc(
        (size_t)corpus->num_movies * vector_dim * sizeof(float)
    );
    float* normalized_query = (float*)malloc(
        vector_dim * sizeof(float)
    );
    MovieResult* results = (MovieResult*)malloc(
        (size_t)(corpus->num_movies - 1) * sizeof(MovieResult)
    );
    if (!normalized_vectors || !normalized_query || !results) {
        fprintf(stderr, "Allocation failed\n");
        free(normalized_vectors);
        free(normalized_query);
        free(results);
        free_movie_corpus(corpus);
        return 1;
    }

    /* Extra buffers for the partitioned mode only; free(NULL) is harmless otherwise. */
    size_t candidates = (size_t)corpus->num_movies - 1;
    CandidateRange* parts = NULL;
    MovieResult* topk = NULL;
    MovieResult* scratch = NULL;
    MovieResult* work = NULL;
    MovieResult* expected = NULL;
    if (mode == MODE_PARTITIONED) {
        parts = (CandidateRange*)malloc((size_t)partition_count * sizeof(*parts));
        topk = (MovieResult*)malloc((size_t)k * sizeof(*topk));
        scratch = (MovieResult*)malloc((size_t)partition_count * (size_t)k * sizeof(*scratch));
        work = (MovieResult*)malloc(candidates * sizeof(*work));
        expected = (MovieResult*)malloc((size_t)k * sizeof(*expected));
        if (!parts || !topk || !scratch || !work || !expected) {
            fprintf(stderr, "Allocation failed\n");
            free(parts);
            free(topk);
            free(scratch);
            free(work);
            free(expected);
            free(normalized_vectors);
            free(normalized_query);
            free(results);
            free_movie_corpus(corpus);
            return 1;
        }
    }
    const MovieResult* shown = results;

    /* Candidate normalization is a one-time cost outside the query timer. */
    double prep_start = get_time_ms();
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        if (i != query) {
            const float* original_vector = corpus->vectors + (size_t)i * vector_dim;
            float* normalized_vector = normalized_vectors + (size_t)i * vector_dim;
            normalize_movie(original_vector, normalized_vector, vector_dim);
        }
    }
    double preparation_ms = get_time_ms() - prep_start;
    double scalar_median_ms = 0.0;
    double simd_median_ms = 0.0;
    double partitioned_median_ms = 0.0;
    double run_times[RUNS];

    /* 1. Scalar Scoring Loop (when mode is scalar or both) */
    if (mode == MODE_SCALAR || mode == MODE_BOTH) {
        for (int run = -1; run < RUNS; run++) {
            double run_start = get_time_ms();
            size_t count = score_candidates(corpus, query, normalized_vectors,
                                            normalized_query, results, 0);
            sort_movie_results(results, count);
            double run_end = get_time_ms();
            if (run >= 0)
                run_times[run] = run_end - run_start;
        }
        qsort(run_times, RUNS, sizeof(double), compare_double);
        scalar_median_ms = (run_times[RUNS / 2 - 1] + run_times[RUNS / 2]) / 2.0;
    }

    /* 2. Single-Thread SIMD Scoring Loop (when mode is simd or both) */
    if (mode == MODE_SIMD || mode == MODE_BOTH) {
        for (int run = -1; run < RUNS; run++) {
            double run_start = get_time_ms();
            size_t count = score_candidates(corpus, query, normalized_vectors,
                                            normalized_query, results, 1);
            sort_movie_results(results, count);
            double run_end = get_time_ms();
            if (run >= 0)
                run_times[run] = run_end - run_start;
        }
        qsort(run_times, RUNS, sizeof(double), compare_double);
        simd_median_ms = (run_times[RUNS / 2 - 1] + run_times[RUNS / 2]) / 2.0;
    }

    /* 3. Serial partitioned top-K with scalar scoring (when mode is partitioned).
     * Scoring, partitioning, per-partition heap selection and merging are all
     * inside the timed region. */
    if (mode == MODE_PARTITIONED) {
        size_t selected = 0;
        for (int run = -1; run < RUNS; run++) {
            double run_start = get_time_ms();
            size_t count = score_candidates(corpus, query, normalized_vectors,
                                            normalized_query, results, 0);
            size_t num_parts = make_even_partitions(count, (size_t)partition_count, parts);
            selected = select_top_k_partitioned(results, count, parts, num_parts,
                                                (size_t)k, scratch, topk);
            double run_end = get_time_ms();
            if (run >= 0)
                run_times[run] = run_end - run_start;
        }
        qsort(run_times, RUNS, sizeof(double), compare_double);
        partitioned_median_ms = (run_times[RUNS / 2 - 1] + run_times[RUNS / 2]) / 2.0;

        /* Untimed check against the full-sort reference on the same score array. */
        size_t want = select_top_k_reference(results, candidates, (size_t)k, work, expected);
        int ok = (selected == want);
        for (size_t i = 0; ok && i < want; i++)
            ok = (topk[i].id == expected[i].id && topk[i].score == expected[i].score);
        if (!ok) {
            fprintf(stderr, "Partitioned result differs from full-sort reference\n");
            free(parts);
            free(topk);
            free(scratch);
            free(work);
            free(expected);
            free(normalized_vectors);
            free(normalized_query);
            free(results);
            free_movie_corpus(corpus);
            return 1;
        }
        shown = topk;
    }

    printf("movies=%u dimension=%u query=%" PRIu64 " K=%" PRIu64 " runs=%d warmups=1\n",
           corpus->num_movies, corpus->vector_dim, id, k, RUNS);

    if (mode == MODE_SCALAR) {
        printf("candidate_normalization_ms=%.6f query_median_ms=%.6f\n",
               preparation_ms, scalar_median_ms);
    } else if (mode == MODE_SIMD) {
        printf("candidate_normalization_ms=%.6f query_median_ms=%.6f\n",
               preparation_ms, simd_median_ms);
    } else if (mode == MODE_PARTITIONED) {
        printf("candidate_normalization_ms=%.6f query_median_ms=%.6f partitions=%" PRIu64 " verify=ok\n",
               preparation_ms, partitioned_median_ms, partition_count);
    } else {
        printf("candidate_normalization_ms=%.6f query_median_ms=%.6f scalar_median_ms=%.6f simd_median_ms=%.6f",
               preparation_ms, simd_median_ms, scalar_median_ms, simd_median_ms);
        if (simd_median_ms > 0.0) {
            printf(" speedup=%.2fx", scalar_median_ms / simd_median_ms);
        }
        putchar('\n');
    }

    for (size_t i = 0; i < (size_t)k; i++) {
        printf("result\t%" PRIu64 "\t", shown[i].id);
        const char* title = corpus->titles[shown[i].index];
        print_title(title);
        printf("\t%.9g\n", shown[i].score);
    }
    free(parts);
    free(topk);
    free(scratch);
    free(work);
    free(expected);
    free(normalized_vectors);
    free(normalized_query);
    free(results);
    free_movie_corpus(corpus);
    return 0;
}