#include "movie_loader.h"
#include "movie_search.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define RUNS 30

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

static int compare_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static int parse_positive(const char* text, uint64_t* value) {
    char* end;
    if (*text < '0' || *text > '9') return 0;
    errno = 0;
    unsigned long long n = strtoull(text, &end, 10);
    if (errno || *end || n == 0) return 0;
    *value = (uint64_t)n;
    return 1;
}

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
        if (corpus->wiki_ids[i] == id) query = i;
    }
    if (query == corpus->num_movies || k >= corpus->num_movies) {
        fprintf(stderr, "Unknown query ID or K outside 1..N-1\n");
        free_movie_corpus(corpus);
        return 1;
    }
    size_t dim = corpus->vector_dim;
    float* normalized = (float*)malloc((size_t)corpus->num_movies * dim * sizeof(float));
    float* q = (float*)malloc(dim * sizeof(float));
    MovieResult* results = (MovieResult*)malloc((size_t)(corpus->num_movies - 1) * sizeof(MovieResult));
    if (!normalized || !q || !results) {
        fprintf(stderr, "Allocation failed\n");
        free(normalized); free(q); free(results); free_movie_corpus(corpus);
        return 1;
    }
    double start = now_seconds();
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        if (i != query) normalize_movie(corpus->vectors + (size_t)i * dim, normalized + (size_t)i * dim, (uint32_t)dim);
    }
    double preparation = now_seconds() - start;
    double times[RUNS];
    for (int run = -1; run < RUNS; run++) {
        start = now_seconds();
        normalize_movie(corpus->vectors + (size_t)query * dim, q, (uint32_t)dim);
        size_t count = 0;
        for (uint32_t i = 0; i < corpus->num_movies; i++) {
            if (i == query) continue;
            float score = 0.0f;
            for (size_t d = 0; d < dim; d++) score += q[d] * normalized[(size_t)i * dim + d];
            results[count].id = corpus->wiki_ids[i];
            results[count].index = i;
            results[count++].score = score;
        }
        sort_movie_results(results, count);
        if (run >= 0) times[run] = now_seconds() - start;
    }
    qsort(times, RUNS, sizeof(double), compare_double);
    printf("movies=%u dimension=%u query=%" PRIu64 " K=%" PRIu64 " runs=%d warmups=1\n", corpus->num_movies, corpus->vector_dim, id, k, RUNS);
    printf("candidate_normalization_seconds=%.9f query_median_seconds=%.9f\n", preparation, (times[RUNS / 2 - 1] + times[RUNS / 2]) / 2.0);
    for (size_t i = 0; i < (size_t)k; i++) {
        printf("result\t%" PRIu64 "\t%s\t%.9g\n", results[i].id, corpus->titles[results[i].index], results[i].score);
    }
    free(normalized); free(q); free(results); free_movie_corpus(corpus);
    return 0;
}
