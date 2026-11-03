#define _POSIX_C_SOURCE 200112L
#include "movie_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

/* 32-byte aligned memory allocation for AVX2 SIMD operations */
static void* aligned_alloc_32(size_t size) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, 32);
#elif defined(__APPLE__) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
    void* ptr = NULL;
    if (posix_memalign(&ptr, 32, size) != 0) {
        return NULL;
    }
    return ptr;
#else
    return malloc(size);
#endif
}

static void aligned_free_32(void* ptr) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int compare_ids(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

CorpusStatus load_movie_corpus(const char* filepath, uint32_t expected_dim, MovieCorpus** out_corpus) {
    if (!filepath || !out_corpus) {
        return CORPUS_ERR_FILE_OPEN;
    }
    *out_corpus = NULL;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return CORPUS_ERR_FILE_OPEN;
    }

    /* 1. Read and validate header */
    MovieHeader header;
    if (fread(&header, sizeof(MovieHeader), 1, f) != 1) {
        fclose(f);
        return CORPUS_ERR_FILE_SIZE_MISMATCH;
    }

    if (memcmp(header.magic, MOVIE_CONTRACT_MAGIC, MOVIE_CONTRACT_MAGIC_LEN) != 0) {
        fclose(f);
        return CORPUS_ERR_INVALID_MAGIC;
    }

    if (header.num_movies == 0) {
        fclose(f);
        return CORPUS_ERR_EMPTY_CORPUS;
    }

    if (header.vector_dim == 0 || (expected_dim > 0 && header.vector_dim != expected_dim)) {
        fclose(f);
        return CORPUS_ERR_DIMENSION_MISMATCH;
    }

    /* 2. Validate exact file size */
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return CORPUS_ERR_FILE_OPEN;
    }
    long actual_file_size = ftell(f);
    if (actual_file_size < 0) {
        fclose(f);
        return CORPUS_ERR_FILE_OPEN;
    }

    if ((size_t)header.vector_dim > (SIZE_MAX - sizeof(MovieMetadataRecord)) / sizeof(float) ||
        (size_t)header.num_movies > (SIZE_MAX - sizeof(MovieHeader)) /
            (sizeof(MovieMetadataRecord) + (size_t)header.vector_dim * sizeof(float))) {
        fclose(f);
        return CORPUS_ERR_FILE_SIZE_MISMATCH;
    }
    size_t expected_size = corpus_expected_file_size(header.num_movies, header.vector_dim);
    if ((size_t)actual_file_size != expected_size) {
        fclose(f);
        return CORPUS_ERR_FILE_SIZE_MISMATCH;
    }

    /* Return to start of metadata (right after header) */
    if (fseek(f, (long)sizeof(MovieHeader), SEEK_SET) != 0) {
        fclose(f);
        return CORPUS_ERR_FILE_OPEN;
    }

    /* 3. Allocate MovieCorpus container */
    MovieCorpus* corpus = (MovieCorpus*)calloc(1, sizeof(MovieCorpus));
    if (!corpus) {
        fclose(f);
        return CORPUS_ERR_ALLOCATION_FAILED;
    }
    corpus->num_movies = header.num_movies;
    corpus->vector_dim = header.vector_dim;

    corpus->wiki_ids = (uint64_t*)malloc(sizeof(uint64_t) * header.num_movies);
    corpus->titles = (char(*)[MOVIE_CONTRACT_MAX_TITLE])malloc(sizeof(char[MOVIE_CONTRACT_MAX_TITLE]) * header.num_movies);
    size_t total_floats = (size_t)header.num_movies * (size_t)header.vector_dim;
    corpus->vectors = (float*)aligned_alloc_32(total_floats * sizeof(float));

    if (!corpus->wiki_ids || !corpus->titles || !corpus->vectors) {
        fclose(f);
        free_movie_corpus(corpus);
        return CORPUS_ERR_ALLOCATION_FAILED;
    }

    /* 4. Read metadata records */
    for (uint32_t i = 0; i < header.num_movies; i++) {
        MovieMetadataRecord rec;
        if (fread(&rec, sizeof(MovieMetadataRecord), 1, f) != 1) {
            fclose(f);
            free_movie_corpus(corpus);
            return CORPUS_ERR_FILE_SIZE_MISMATCH;
        }
        if (rec.wiki_id == 0 || rec.title[0] == '\0') {
            fclose(f);
            free_movie_corpus(corpus);
            return CORPUS_ERR_INVALID_METADATA;
        }
        corpus->wiki_ids[i] = rec.wiki_id;
        memcpy(corpus->titles[i], rec.title, MOVIE_CONTRACT_MAX_TITLE);
        corpus->titles[i][MOVIE_CONTRACT_MAX_TITLE - 1] = '\0'; /* Ensure null-terminated */
    }

    uint64_t* sorted_ids = (uint64_t*)malloc(sizeof(uint64_t) * header.num_movies);
    if (!sorted_ids) {
        fclose(f);
        free_movie_corpus(corpus);
        return CORPUS_ERR_ALLOCATION_FAILED;
    }
    memcpy(sorted_ids, corpus->wiki_ids, sizeof(uint64_t) * header.num_movies);
    qsort(sorted_ids, header.num_movies, sizeof(uint64_t), compare_ids);
    for (uint32_t i = 1; i < header.num_movies; i++) {
        if (sorted_ids[i] == sorted_ids[i - 1]) {
            free(sorted_ids);
            fclose(f);
            free_movie_corpus(corpus);
            return CORPUS_ERR_INVALID_METADATA;
        }
    }
    free(sorted_ids);

    /* 5. Read vector buffer */
    if (fread(corpus->vectors, sizeof(float), total_floats, f) != total_floats) {
        fclose(f);
        free_movie_corpus(corpus);
        return CORPUS_ERR_FILE_SIZE_MISMATCH;
    }
    fclose(f);

    /* 6. Validate vector values (Finite check and Zero-Magnitude check) */
    for (uint32_t i = 0; i < header.num_movies; i++) {
        const float* vec = corpus->vectors + ((size_t)i * header.vector_dim);
        double sum_sq = 0.0;

        for (uint32_t d = 0; d < header.vector_dim; d++) {
            float val = vec[d];
            if (isnan(val) || isinf(val)) {
                free_movie_corpus(corpus);
                return CORPUS_ERR_NON_FINITE_VALUE;
            }
            sum_sq += (double)val * (double)val;
        }

        double norm = sqrt(sum_sq);
        if (norm < (double)MOVIE_CONTRACT_MIN_NORM) {
            free_movie_corpus(corpus);
            return CORPUS_ERR_ZERO_MAGNITUDE;
        }
    }

    *out_corpus = corpus;
    return CORPUS_OK;
}

void free_movie_corpus(MovieCorpus* corpus) {
    if (!corpus) return;
    if (corpus->wiki_ids) {
        free(corpus->wiki_ids);
        corpus->wiki_ids = NULL;
    }
    if (corpus->titles) {
        free(corpus->titles);
        corpus->titles = NULL;
    }
    if (corpus->vectors) {
        aligned_free_32(corpus->vectors);
        corpus->vectors = NULL;
    }
    free(corpus);
}
