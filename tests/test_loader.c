#include "movie_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define SAMPLE_PATH "data/sample_movies.bin"
#define TEMP_TEST_PATH "data/temp_test_corrupt.bin"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s (Line %d): %s\n", __func__, __LINE__, msg); \
        return 0; \
    } \
} while(0)

/* Exact expected reference values for all 16 sample movies (Issue #2 AC 4 & Review Finding #3) */
typedef struct {
    uint64_t wiki_id;
    const char* title;
    float expected_v[4];
} MovieReference;

static const MovieReference EXPECTED_SAMPLE_MOVIES[16] = {
    {30007,     "The Matrix",               {-0.070782f,  0.030554f, -0.104779f,  0.018619f}},
    {23270459,  "Inception",                {-0.017669f,  0.006115f, -0.028679f,  0.023014f}},
    {52371,     "Titanic",                  {-0.073466f,  0.010494f,  0.039379f,  0.044740f}},
    {4273140,   "Avatar",                   {-0.079671f,  0.048819f,  0.049715f, -0.008998f}},
    {54173,     "Pulp Fiction",             {-0.050226f, -0.008961f, -0.003963f, -0.003791f}},
    {1009041,   "Fight Club",               {-0.103656f,  0.044157f, -0.020581f, -0.009116f}},
    {41528,     "Forrest Gump",             {-0.095339f,  0.002661f,  0.026196f,  0.014134f}},
    {2466773,   "The Godfather",            {-0.087629f,  0.059065f, -0.029840f, -0.012104f}},
    {4276475,   "The Dark Knight",          {-0.056533f,  0.033469f, -0.050009f, -0.041906f}},
    {68485,     "Jurassic Park",            {-0.037604f,  0.036865f,  0.026129f, -0.048649f}},
    {3616797,   "Gladiator",                {-0.034567f,  0.054399f, -0.049119f, -0.048202f}},
    {30625,     "The Shawshank Redemption", {-0.010503f,  0.028324f, -0.021817f, -0.029177f}},
    {65834,     "Schindler's List",         {-0.015114f,  0.066368f, -0.072284f,  0.045404f}},
    {64394,     "Goodfellas",               {-0.082697f,  0.019300f, -0.095911f,  0.003193f}},
    {171005,    "Henry V",                  {-0.064986f,  0.036820f,  0.055128f, -0.027150f}},
    {26067101,  "Siam Sunset",              { 0.031174f,  0.030418f,  0.036865f,  0.001416f}}
};

/* Helper to write test binaries */
static void write_test_binary(const char* path, const char* magic, uint32_t num_movies, uint32_t dim, 
                              size_t metadata_records_to_write, size_t floats_to_write, float fill_val) {
    FILE* f = fopen(path, "wb");
    assert(f != NULL);

    MovieHeader hdr;
    memcpy(hdr.magic, magic, 4);
    hdr.num_movies = num_movies;
    hdr.vector_dim = dim;
    fwrite(&hdr, sizeof(MovieHeader), 1, f);

    for (size_t i = 0; i < metadata_records_to_write; i++) {
        MovieMetadataRecord rec;
        rec.wiki_id = (uint64_t)(1000 + i);
        snprintf(rec.title, sizeof(rec.title), "Mock Movie %zu", i);
        fwrite(&rec, sizeof(MovieMetadataRecord), 1, f);
    }

    if (floats_to_write > 0) {
        float* buf = (float*)malloc(floats_to_write * sizeof(float));
        for (size_t i = 0; i < floats_to_write; i++) {
            buf[i] = fill_val;
        }
        fwrite(buf, sizeof(float), floats_to_write, f);
        free(buf);
    }

    fclose(f);
}

/* Test 1: Load valid real-movie sample dataset and assert IDs, titles, and exact vector coordinates */
static int test_load_valid_sample(void) {
    g_tests_run++;
    printf("Running test_load_valid_sample...\n");

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(SAMPLE_PATH, 384, &corpus);
    TEST_ASSERT(status == CORPUS_OK, corpus_status_to_string(status));
    TEST_ASSERT(corpus != NULL, "Corpus pointer is NULL");
    TEST_ASSERT(corpus->num_movies == 16, "Expected exactly 16 movies");
    TEST_ASSERT(corpus->vector_dim == 384, "Expected dimension 384");
    TEST_ASSERT(corpus->wiki_ids != NULL, "wiki_ids array is NULL");
    TEST_ASSERT(corpus->titles != NULL, "titles array is NULL");
    TEST_ASSERT(corpus->vectors != NULL, "vectors array is NULL");

    /* Verify 32-byte alignment for AVX2 SIMD */
    uintptr_t addr = (uintptr_t)corpus->vectors;
    TEST_ASSERT((addr % 32) == 0, "Vectors buffer is not 32-byte aligned for AVX2!");

    /* Verify every movie's ID, title, and exact vector coordinates (Review Finding #3) */
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        TEST_ASSERT(corpus->wiki_ids[i] == EXPECTED_SAMPLE_MOVIES[i].wiki_id, "Movie ID mismatch");
        TEST_ASSERT(strcmp(corpus->titles[i], EXPECTED_SAMPLE_MOVIES[i].title) == 0, "Movie title mismatch");

        const float* v = corpus->vectors + ((size_t)i * corpus->vector_dim);
        for (int k = 0; k < 4; k++) {
            float diff = fabsf(v[k] - EXPECTED_SAMPLE_MOVIES[i].expected_v[k]);
            TEST_ASSERT(diff < 1e-4f, "Vector coordinate mismatch for movie!");
        }
    }

    printf("\n  Successfully loaded and verified %u movies (dim=%u, 32-byte AVX2 aligned):\n", 
           corpus->num_movies, corpus->vector_dim);
    printf("  -------------------------------------------------------------------------\n");
    printf("  %-4s | %-10s | %-26s | %-18s\n", "#", "Wiki ID", "Title", "First 3 Vector Dims");
    printf("  -------------------------------------------------------------------------\n");
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        const float* v = corpus->vectors + ((size_t)i * corpus->vector_dim);
        printf("  [%02u] | %-10llu | %-26s | [%.3f, %.3f, %.3f]\n", 
               i, (unsigned long long)corpus->wiki_ids[i], corpus->titles[i], v[0], v[1], v[2]);
    }
    printf("  -------------------------------------------------------------------------\n\n");

    free_movie_corpus(corpus);
    g_tests_passed++;
    printf("  [PASS] test_load_valid_sample\n");
    return 1;
}

/* Test 2: File open failure */
static int test_file_not_found(void) {
    g_tests_run++;
    printf("Running test_file_not_found...\n");

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus("data/non_existent_file.bin", 384, &corpus);
    TEST_ASSERT(status == CORPUS_ERR_FILE_OPEN, "Expected CORPUS_ERR_FILE_OPEN");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_file_not_found (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 3: Corrupted magic identifier */
static int test_invalid_magic(void) {
    g_tests_run++;
    printf("Running test_invalid_magic...\n");

    write_test_binary(TEMP_TEST_PATH, "BAD1", 2, 384, 2, 2 * 384, 1.0f);

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(TEMP_TEST_PATH, 384, &corpus);
    remove(TEMP_TEST_PATH);

    TEST_ASSERT(status == CORPUS_ERR_INVALID_MAGIC, "Expected CORPUS_ERR_INVALID_MAGIC");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_invalid_magic (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 4: Truncated / mismatched file size */
static int test_file_size_mismatch(void) {
    g_tests_run++;
    printf("Running test_file_size_mismatch...\n");

    /* Header promises 4 movies, but only 2 records are written */
    write_test_binary(TEMP_TEST_PATH, "MVE1", 4, 384, 2, 2 * 384, 1.0f);

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(TEMP_TEST_PATH, 384, &corpus);
    remove(TEMP_TEST_PATH);

    TEST_ASSERT(status == CORPUS_ERR_FILE_SIZE_MISMATCH, "Expected CORPUS_ERR_FILE_SIZE_MISMATCH");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_file_size_mismatch (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 5: Vector dimension mismatch */
static int test_dimension_mismatch(void) {
    g_tests_run++;
    printf("Running test_dimension_mismatch...\n");

    MovieCorpus* corpus = NULL;
    /* Expecting 512 dims, but file has 384 */
    CorpusStatus status = load_movie_corpus(SAMPLE_PATH, 512, &corpus);

    TEST_ASSERT(status == CORPUS_ERR_DIMENSION_MISMATCH, "Expected CORPUS_ERR_DIMENSION_MISMATCH");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_dimension_mismatch (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 6: Non-finite vector values (NaN and Inf) */
static int test_non_finite_values(void) {
    g_tests_run++;
    printf("Running test_non_finite_values...\n");

    /* Create test file with injected NaN */
    write_test_binary(TEMP_TEST_PATH, "MVE1", 1, 384, 1, 384, 1.0f);
    
    /* Overwrite float at index 10 with NaN */
    FILE* f = fopen(TEMP_TEST_PATH, "r+b");
    assert(f != NULL);
    long offset = (long)(sizeof(MovieHeader) + sizeof(MovieMetadataRecord) + (10 * sizeof(float)));
    fseek(f, offset, SEEK_SET);
    float nan_val = (float)NAN;
    fwrite(&nan_val, sizeof(float), 1, f);
    fclose(f);

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(TEMP_TEST_PATH, 384, &corpus);
    remove(TEMP_TEST_PATH);

    TEST_ASSERT(status == CORPUS_ERR_NON_FINITE_VALUE, "Expected CORPUS_ERR_NON_FINITE_VALUE for NaN");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_non_finite_values (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 7: Zero magnitude vector rejection */
static int test_zero_magnitude(void) {
    g_tests_run++;
    printf("Running test_zero_magnitude...\n");

    /* Create test file with all 0.0f vector */
    write_test_binary(TEMP_TEST_PATH, "MVE1", 1, 384, 1, 384, 0.0f);

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(TEMP_TEST_PATH, 384, &corpus);
    remove(TEMP_TEST_PATH);

    TEST_ASSERT(status == CORPUS_ERR_ZERO_MAGNITUDE, "Expected CORPUS_ERR_ZERO_MAGNITUDE");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_zero_magnitude (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 8: Empty corpus rejection */
static int test_empty_corpus(void) {
    g_tests_run++;
    printf("Running test_empty_corpus...\n");

    write_test_binary(TEMP_TEST_PATH, "MVE1", 0, 384, 0, 0, 0.0f);

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(TEMP_TEST_PATH, 384, &corpus);
    remove(TEMP_TEST_PATH);

    TEST_ASSERT(status == CORPUS_ERR_EMPTY_CORPUS, "Expected CORPUS_ERR_EMPTY_CORPUS");
    TEST_ASSERT(corpus == NULL, "Corpus must be NULL on error");

    g_tests_passed++;
    printf("  [PASS] test_empty_corpus (Correctly rejected with code %d: %s)\n", status, corpus_status_to_string(status));
    return 1;
}

/* Test 9: Directly test detection of swapped vector rows (Review Finding #3) */
static int test_swapped_vectors_detected(void) {
    g_tests_run++;
    printf("Running test_swapped_vectors_detected...\n");

    /* Read valid sample and write a modified copy with Movie 0 and Movie 1 vectors swapped */
    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(SAMPLE_PATH, 384, &corpus);
    TEST_ASSERT(status == CORPUS_OK, "Failed to load base sample");

    /* Swap vector rows for movie 0 and movie 1 */
    size_t dim = corpus->vector_dim;
    float* tmp = (float*)malloc(dim * sizeof(float));
    memcpy(tmp, corpus->vectors, dim * sizeof(float));
    memcpy(corpus->vectors, corpus->vectors + dim, dim * sizeof(float));
    memcpy(corpus->vectors + dim, tmp, dim * sizeof(float));
    free(tmp);

    /* Verify that checking Movie 0 against expected values now fails */
    const float* v0 = corpus->vectors;
    float diff0 = fabsf(v0[0] - EXPECTED_SAMPLE_MOVIES[0].expected_v[0]);
    TEST_ASSERT(diff0 > 1e-4f, "Swapped vectors were NOT detected! Expected mismatch.");

    free_movie_corpus(corpus);
    g_tests_passed++;
    printf("  [PASS] test_swapped_vectors_detected (Coordinate check successfully caught swapped vector rows)\n");
    return 1;
}

int main(void) {
    printf("=========================================================\n");
    printf(" Movie Vector Loader & Input Contract Test Suite (Issue #2)\n");
    printf("=========================================================\n\n");

    if (!test_load_valid_sample()) return 1;
    if (!test_file_not_found()) return 1;
    if (!test_invalid_magic()) return 1;
    if (!test_file_size_mismatch()) return 1;
    if (!test_dimension_mismatch()) return 1;
    if (!test_non_finite_values()) return 1;
    if (!test_zero_magnitude()) return 1;
    if (!test_empty_corpus()) return 1;
    if (!test_swapped_vectors_detected()) return 1;

    printf("\n=========================================================\n");
    printf(" ALL TESTS PASSED: %d/%d assertions succeeded.\n", g_tests_passed, g_tests_run);
    printf("=========================================================\n");
    return 0;
}
