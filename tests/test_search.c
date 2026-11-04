/** @file
 * @brief End-to-end checks for scalar movie search and input validation.
 */
#include "movie_contract.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#include <process.h>
#define make_dir(path) _mkdir(path)
#define remove_dir(path) _rmdir(path)
#define process_id() _getpid()
#define SEARCH_COMMAND "tests\\movie_search.exe"
#else
#include <sys/stat.h>
#include <unistd.h>
#define make_dir(path) mkdir(path, 0700)
#define remove_dir(path) rmdir(path)
#define process_id() getpid()
#define SEARCH_COMMAND "tests/movie_search"
#endif

/* Scores in this small, three-dimensional fixture must agree within 1e-5. */
#define SCORE_TOLERANCE 1e-5
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL at line %d: %s\n", __LINE__, #condition); \
        goto cleanup; \
    } \
} while (0)

static const uint64_t ids[] = {50, 30, 40, 10, 20, 60};
static const char* titles[] = {"Query", "Same", "Tie high", "Tie low", "Opposite", "Orthogonal"};
static const float vectors[][3] = {
    {3, 4, 0}, {6, 8, 0}, {0, 2, 0}, {0, 1, 0}, {-3, -4, 0}, {4, -3, 0}
};

/**
 * Computes a reference cosine directly in double precision.
 * @param a First three-dimensional vector.
 * @param b Second three-dimensional vector.
 * @return Cosine similarity of the nonzero input vectors.
 */
static double reference_score(const float* a, const float* b) {
    double dot = 0, aa = 0, bb = 0;
    for (int i = 0; i < 3; i++) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return dot / (sqrt(aa) * sqrt(bb));
}

/**
 * Writes a search fixture, optionally changing one field or vector.
 * @param path Destination binary file.
 * @param variant 0 for valid data; 1 through 9 select the test variants below.
 * @return 1 if all writes and the close succeed, otherwise 0.
 */
static int write_fixture(const char* path, int variant) {
    MovieHeader header = {{'M', 'V', 'E', '1'}, variant == 1 ? 1 : 6, variant == 2 ? 0 : 3};
    FILE* file = fopen(path, "wb");
    if (!file)
        return 0;
    int ok = fwrite(&header, sizeof(header), 1, file) == 1;
    for (uint32_t i = 0; i < header.num_movies; i++) {
        MovieMetadataRecord record = {ids[i], ""};
        if (variant == 3 && i == 1)
            record.wiki_id = ids[0];
        if (variant == 4 && i == 1)
            record.wiki_id = 0;
        if (variant != 5 || i != 1)
            strcpy(record.title, titles[i]);
        ok &= fwrite(&record, sizeof(record), 1, file) == 1;
    }
    if (header.vector_dim) {
        for (uint32_t i = 0; i < header.num_movies; i++) {
            float v[3];
            memcpy(v, vectors[i], sizeof(v));
            if (variant == 6 && i == 1)
                v[0] = v[1] = v[2] = 0;
            if (variant == 7 && i == 1)
                v[0] = NAN;
            if (variant == 8 && i == 1)
                v[0] = INFINITY;
            if (variant == 9 && i == 1)
                v[0] = v[1] = FLT_MAX;
            ok &= fwrite(v, sizeof(v), 1, file) == 1;
        }
    }
    return fclose(file) == 0 && ok;
}

/**
 * Checks ranking, scores, argument validation, and corpus validation.
 * @return 0 if all checks pass, otherwise 1.
 */
int main(void) {
    char dir[80], fixture[100], output[100], command[300], line[512], title[128];
    /* Same direction, tied scores ordered by ID, orthogonal, then opposite. */
    const int expected[] = {1, 3, 2, 5, 4};
    int passed = 0;
    FILE* file = NULL;
    snprintf(dir, sizeof(dir), "tests/search_check_%ld", (long)process_id());
    if (make_dir(dir) != 0) {
        fprintf(stderr, "Cannot create test directory %s\n", dir);
        return 1;
    }
    snprintf(fixture, sizeof(fixture), "%s/input.bin", dir);
    snprintf(output, sizeof(output), "%s/output.txt", dir);
    CHECK(write_fixture(fixture, 0));
    snprintf(command, sizeof(command), SEARCH_COMMAND " %s 50 5 > %s 2>&1", fixture, output);
    CHECK(system(command) == 0);
    file = fopen(output, "r");
    CHECK(file != NULL);
    CHECK(fgets(line, sizeof(line), file) != NULL);
    CHECK(fgets(line, sizeof(line), file) != NULL);
    double preparation, median;
    CHECK(sscanf(line, "candidate_normalization_ms=%lf query_median_ms=%lf", &preparation, &median) == 2);
    CHECK(isfinite(preparation) && preparation >= 0 && isfinite(median) && median >= 0);
    for (int i = 0; i < 5; i++) {
        unsigned long long id;
        double score;
        int index = expected[i];
        CHECK(fgets(line, sizeof(line), file) != NULL);
        CHECK(sscanf(line, "result\t%llu\t%127[^\t]\t%lf", &id, title, &score) == 3);
        CHECK(id == ids[index] && strcmp(title, titles[index]) == 0);
        CHECK(isfinite(score) && fabs(score - reference_score(vectors[0], vectors[index])) <= SCORE_TOLERANCE);
    }
    CHECK(fgets(line, sizeof(line), file) == NULL);
    CHECK(fclose(file) == 0);
    file = NULL;

    snprintf(command, sizeof(command), SEARCH_COMMAND " %s 50 2 > %s 2>&1", fixture, output);
    CHECK(system(command) == 0);
    file = fopen(output, "r");
    CHECK(file != NULL);
    CHECK(fgets(line, sizeof(line), file) != NULL && fgets(line, sizeof(line), file) != NULL);
    for (int i = 0; i < 2; i++) {
        unsigned long long id;
        CHECK(fgets(line, sizeof(line), file) != NULL);
        CHECK(sscanf(line, "result\t%llu", &id) == 1 && id == ids[expected[i]]);
    }
    CHECK(fgets(line, sizeof(line), file) == NULL);
    CHECK(fclose(file) == 0);
    file = NULL;

    const char* invalid_args[] = {"50 0", "50 6", "999 1", "50 -1", "50 1x", "50 18446744073709551616"};
    for (size_t i = 0; i < sizeof(invalid_args) / sizeof(invalid_args[0]); i++) {
        snprintf(command, sizeof(command), SEARCH_COMMAND " %s %s > %s 2>&1", fixture, invalid_args[i], output);
        CHECK(system(command) != 0);
        file = fopen(output, "r");
        CHECK(file != NULL && fgets(line, sizeof(line), file) != NULL);
        CHECK(strstr(line, i == 1 || i == 2 ? "Unknown query ID or K" : "Usage:") != NULL);
        CHECK(fclose(file) == 0);
        file = NULL;
    }
    for (int variant = 1; variant <= 8; variant++) {
        CHECK(write_fixture(fixture, variant));
        snprintf(command, sizeof(command), SEARCH_COMMAND " %s 50 1 > %s 2>&1", fixture, output);
        CHECK(system(command) != 0);
        file = fopen(output, "r");
        CHECK(file != NULL && fgets(line, sizeof(line), file) != NULL);
        CHECK(strstr(line, variant == 1 ? "Unknown query ID or K" : "Invalid corpus:") != NULL);
        CHECK(fclose(file) == 0);
        file = NULL;
    }
    CHECK(write_fixture(fixture, 9));
    snprintf(command, sizeof(command), SEARCH_COMMAND " %s 50 5 > %s 2>&1", fixture, output);
    CHECK(system(command) == 0);
    file = fopen(output, "r");
    CHECK(file != NULL);
    CHECK(fgets(line, sizeof(line), file) != NULL && fgets(line, sizeof(line), file) != NULL);
    const float large[] = {FLT_MAX, FLT_MAX, 0};
    int found = 0;
    while (fgets(line, sizeof(line), file)) {
        unsigned long long id;
        double score;
        CHECK(sscanf(line, "result\t%llu\t%127[^\t]\t%lf", &id, title, &score) == 3);
        if (id == 30) {
            CHECK(isfinite(score) && fabs(score - reference_score(vectors[0], large)) <= SCORE_TOLERANCE);
            found++;
        }
    }
    CHECK(found == 1);
    passed = 1;
cleanup:
    if (file)
        fclose(file);
    remove(fixture);
    remove(output);
    remove_dir(dir);
    if (passed)
        puts("Search correctness checks passed.");
    return passed ? 0 : 1;
}
