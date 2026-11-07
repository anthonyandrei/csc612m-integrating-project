#include "movie_contract.h"
#include "movie_loader.h"
#include "movie_search.h"
#include "simd_support.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "[FAIL] Line %d: Assertion '%s' failed!\n", __LINE__, #cond); \
        return 1; \
    } \
} while (0)

#define SCORE_TOLERANCE 1e-5

/* Independent reference dot product in double precision */
static double reference_dot(const float* a, const float* b, uint32_t dim) {
    double sum = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

static int test_known_vectors(void) {
    float a[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    float b[8] = {0, 1, 0, 0, 0, 0, 0, 0};
    float c[8] = {-1, 0, 0, 0, 0, 0, 0, 0};

    float dot_ortho = dot_product_simd_ymm(a, b, 8);
    CHECK(fabsf(dot_ortho - 0.0f) < SCORE_TOLERANCE);

    float dot_ident = dot_product_simd_ymm(a, a, 8);
    CHECK(fabsf(dot_ident - 1.0f) < SCORE_TOLERANCE);

    float dot_opp = dot_product_simd_ymm(a, c, 8);
    CHECK(fabsf(dot_opp - (-1.0f)) < SCORE_TOLERANCE);

    printf("  [PASS] test_known_vectors (orthogonal, collinear, opposite)\n");
    return 0;
}

static int test_remainder_dimensions(void) {
    static const uint32_t test_dims[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 383, 384, 385};
    size_t num_tests = sizeof(test_dims) / sizeof(test_dims[0]);

    for (size_t t = 0; t < num_tests; t++) {
        uint32_t dim = test_dims[t];
        if (dim == 0) {
            float zero_res = dot_product_simd_ymm(NULL, NULL, 0);
            CHECK(zero_res == 0.0f);
            continue;
        }

        float* va = (float*)malloc(dim * sizeof(float));
        float* vb = (float*)malloc(dim * sizeof(float));
        CHECK(va && vb);

        for (uint32_t i = 0; i < dim; i++) {
            va[i] = (float)sin((double)i * 0.17 + 0.5);
            vb[i] = (float)cos((double)i * 0.23 + 0.1);
        }

        double ref = reference_dot(va, vb, dim);
        float simd = dot_product_simd_ymm(va, vb, dim);

        CHECK(fabs((double)simd - ref) < (double)SCORE_TOLERANCE * (1.0 + (double)dim * 0.001));

        free(va);
        free(vb);
    }
    printf("  [PASS] test_remainder_dimensions (all remainder sizes match double-precision reference)\n");
    return 0;
}

static int test_sample_corpus_parity(void) {
    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus("data/sample_movies.bin", 0, &corpus);
    CHECK(status == CORPUS_OK);
    CHECK(corpus->num_movies > 1);

    uint32_t dim = corpus->vector_dim;
    float* norm_a = (float*)malloc(dim * sizeof(float));
    float* norm_b = (float*)malloc(dim * sizeof(float));
    CHECK(norm_a && norm_b);

    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        normalize_movie(corpus->vectors + (size_t)i * dim, norm_a, dim);
        for (uint32_t j = 0; j < corpus->num_movies; j++) {
            normalize_movie(corpus->vectors + (size_t)j * dim, norm_b, dim);

            /* Scalar C dot product */
            float scalar_dot = 0.0f;
            for (uint32_t d = 0; d < dim; d++) {
                scalar_dot += norm_a[d] * norm_b[d];
            }

            /* NASM SIMD dot product */
            float simd_dot = dot_product_simd_ymm(norm_a, norm_b, dim);

            /* Double precision reference */
            double ref_dot = reference_dot(norm_a, norm_b, dim);

            CHECK(fabsf(scalar_dot - simd_dot) < SCORE_TOLERANCE);
            CHECK(fabs((double)simd_dot - ref_dot) < SCORE_TOLERANCE);
        }
    }

    free(norm_a);
    free(norm_b);
    free_movie_corpus(corpus);
    printf("  [PASS] test_sample_corpus_parity (sample corpus pairwise scores match perfectly)\n");
    return 0;
}

int main(void) {
    printf("=========================================================\n");
    printf(" Single-Thread SIMD Kernel Verification Suite (Issue #5)\n");
    printf("=========================================================\n");

    SIMDCapabilities caps;
    detect_simd_capabilities(&caps);
    print_simd_capabilities(&caps);

    if (test_known_vectors() != 0) return 1;
    if (test_remainder_dimensions() != 0) return 1;
    if (test_sample_corpus_parity() != 0) return 1;

    printf("\n=========================================================\n");
    printf(" ALL SIMD TESTS PASSED: 100%% precision & parity verified.\n");
    printf("=========================================================\n");
    return 0;
}
