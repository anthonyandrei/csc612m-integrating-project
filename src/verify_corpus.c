#include "movie_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_mve1.bin> [expected_count]\n", argv[0]);
        return 1;
    }

    MovieCorpus* corpus = NULL;
    CorpusStatus status = load_movie_corpus(argv[1], MOVIE_CONTRACT_DEFAULT_DIM, &corpus);
    
    if (status != CORPUS_OK) {
        fprintf(stderr, "FAIL: Loader rejected the file: %s\n", corpus_status_to_string(status));
        return 1;
    }

    /* 1. Check explicit count from manifest (if provided) */
    if (argc >= 3) {
        uint32_t expected = (uint32_t)strtoul(argv[2], NULL, 10);
        if (corpus->num_movies != expected) {
            fprintf(stderr, "FAIL: Manifest count (%u) != Loader count (%u)\n", expected, corpus->num_movies);
            free_movie_corpus(corpus);
            return 1;
        }
    }

    /* 2. Full internal validation of all records */
    for (uint32_t i = 0; i < corpus->num_movies; i++) {
        if (corpus->wiki_ids[i] == 0) {
            fprintf(stderr, "FAIL: Record %u has zero wiki_id\n", i);
            free_movie_corpus(corpus); return 1;
        }
        if (corpus->titles[i][0] == '\0') {
            fprintf(stderr, "FAIL: Record %u has empty title\n", i);
            free_movie_corpus(corpus); return 1;
        }

        const float* v = corpus->vectors + (size_t)i * corpus->vector_dim;
        double ss = 0.0;
        for (uint32_t d = 0; d < corpus->vector_dim; d++) {
            if (!isfinite(v[d])) {
                fprintf(stderr, "FAIL: Record %u has non-finite float\n", i);
                free_movie_corpus(corpus); return 1;
            }
            ss += (double)v[d] * (double)v[d];
        }
        if (sqrt(ss) < 1e-6) {
            fprintf(stderr, "FAIL: Record %u has zero magnitude\n", i);
            free_movie_corpus(corpus); return 1;
        }
    }

    printf("VERIFY OK: %u movies loaded, all IDs/titles/vectors valid.\n", corpus->num_movies);
    free_movie_corpus(corpus);
    return 0;
}