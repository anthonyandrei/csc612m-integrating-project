/** @file
 * @brief Command-line scalar movie search and timing benchmark.
 */
#include "movie_loader.h"
#include "movie_search.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

/** Number of timed searches after the warmup. */
#define RUNS 30

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
 * Runs a fixed-query benchmark and prints its ordered top-K results.
 *
 * Arguments are the corpus path, query movie ID, and number of results.
 * One warmup precedes RUNS measured searches; loading and printing are excluded
 * from query timing. Candidate normalization is timed separately.
 * @param argc Argument count, including the program name; must be 4.
 * @param argv Argument strings: program name, corpus path, query ID, and K.
 * @return 0 on success, or 1 for invalid input or an allocation failure.
 */
int main(int argc, char** argv) {
    uint64_t id, k;
    MovieCorpus* corpus = NULL;
    if (argc != 4 || !parse_positive(argv[2], &id) || !parse_positive(argv[3], &k)) {
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
    /* Candidate normalization is a one-time cost outside the query timer. */
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        if (i != query) {
            const float* original_vector = corpus->vectors + (size_t)i * vector_dim;
            float* normalized_vector = normalized_vectors + (size_t)i * vector_dim;
            normalize_movie(original_vector, normalized_vector, vector_dim);
        }
    }
    QueryPerformanceCounter(&end);
    double preparation_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    double run_times[RUNS];
    /* The first iteration warms up; measured iterations include query normalization,
       scoring, and sorting, but exclude loading and printing. */
    for (int run = -1; run < RUNS; run++) {
        QueryPerformanceCounter(&start);
        const float* query_vector = corpus->vectors + (size_t)query * vector_dim;
        normalize_movie(query_vector, normalized_query, vector_dim);
        size_t count = 0;
        for (uint32_t i = 0; i < corpus->num_movies; i++) {
            if (i == query)
                continue;
            /* The dot product of unit vectors is their cosine similarity. */
            float score = 0.0f;
            for (size_t d = 0; d < vector_dim; d++) {
                score += normalized_query[d] * normalized_vectors[(size_t)i * vector_dim + d];
            }
            results[count].id = corpus->wiki_ids[i];
            results[count].index = i;
            results[count++].score = score;
        }
        sort_movie_results(results, count);
        QueryPerformanceCounter(&end);
        if (run >= 0)
            run_times[run] = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    }
    /* With 30 runs, the median is the average of the two middle durations. */
    qsort(run_times, RUNS, sizeof(double), compare_double);
    double median_ms = (run_times[RUNS / 2 - 1] + run_times[RUNS / 2]) / 2.0;
    printf("movies=%u dimension=%u query=%" PRIu64 " K=%" PRIu64 " runs=%d warmups=1\n", corpus->num_movies, corpus->vector_dim, id, k, RUNS);
    printf("candidate_normalization_ms=%.6f query_median_ms=%.6f\n", preparation_ms, median_ms);
    for (size_t i = 0; i < (size_t)k; i++) {
        printf("result\t%" PRIu64 "\t", results[i].id);
        const char* title = corpus->titles[results[i].index];
        print_title(title);
        printf("\t%.9g\n", results[i].score);
    }
    free(normalized_vectors);
    free(normalized_query);
    free(results);
    free_movie_corpus(corpus);
    return 0;
}
