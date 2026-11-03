#ifndef MOVIE_CONTRACT_H
#define MOVIE_CONTRACT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * Movie Vector Preparation Pipeline - Shared Binary Input Contract (MVE1)
 * ============================================================================
 * This header defines the exact binary format, data structures, and validation
 * contract shared between the offline vector generator and the C search program.
 */

#define MOVIE_CONTRACT_MAGIC        "MVE1"
#define MOVIE_CONTRACT_MAGIC_LEN    4
#define MOVIE_CONTRACT_DEFAULT_DIM  384
#define MOVIE_CONTRACT_MAX_TITLE    128
#define MOVIE_CONTRACT_MIN_NORM     1e-6f

/* Error & Validation Return Codes */
typedef enum {
    CORPUS_OK                       = 0,
    CORPUS_ERR_FILE_OPEN            = 1,
    CORPUS_ERR_INVALID_MAGIC        = 2,
    CORPUS_ERR_FILE_SIZE_MISMATCH   = 3,
    CORPUS_ERR_DIMENSION_MISMATCH   = 4,
    CORPUS_ERR_NON_FINITE_VALUE     = 5,
    CORPUS_ERR_ZERO_MAGNITUDE       = 6,
    CORPUS_ERR_ALLOCATION_FAILED    = 7,
    CORPUS_ERR_EMPTY_CORPUS         = 8,
    CORPUS_ERR_INVALID_METADATA     = 9  /* Zero or duplicate ID, or empty title. */
} CorpusStatus;

#pragma pack(push, 1)

/* File Header: 12 bytes total */
typedef struct {
    char magic[MOVIE_CONTRACT_MAGIC_LEN]; /* Must be 'M', 'V', 'E', '1' */
    uint32_t num_movies;                  /* Number of movie records */
    uint32_t vector_dim;                  /* Vector dimension (nominal: 384) */
} MovieHeader;

/* Metadata Record per Movie: 136 bytes total */
typedef struct {
    uint64_t wiki_id;                     /* Wikipedia Movie ID */
    char title[MOVIE_CONTRACT_MAX_TITLE]; /* Null-terminated UTF-8 movie title */
} MovieMetadataRecord;

#pragma pack(pop)

/* In-memory loaded corpus representation */
typedef struct {
    uint32_t num_movies;
    uint32_t vector_dim;
    uint64_t* wiki_ids;                   /* Array of size num_movies */
    char (*titles)[MOVIE_CONTRACT_MAX_TITLE]; /* Array of size num_movies * 128 */
    float* vectors;                       /* Contiguous array (num_movies * vector_dim), 32-byte aligned */
} MovieCorpus;

/**
 * Calculates the exact expected binary file size for a given movie count and dimension.
 * Expected Size = sizeof(MovieHeader) + (num_movies * sizeof(MovieMetadataRecord)) + (num_movies * vector_dim * sizeof(float))
 */
static inline size_t corpus_expected_file_size(uint32_t num_movies, uint32_t vector_dim) {
    return sizeof(MovieHeader) + 
           ((size_t)num_movies * sizeof(MovieMetadataRecord)) + 
           ((size_t)num_movies * (size_t)vector_dim * sizeof(float));
}

/**
 * Returns a human-readable description of a CorpusStatus code.
 */
static inline const char* corpus_status_to_string(CorpusStatus status) {
    switch (status) {
        case CORPUS_OK:
            return "Success";
        case CORPUS_ERR_FILE_OPEN:
            return "Failed to open file";
        case CORPUS_ERR_INVALID_MAGIC:
            return "Invalid magic identifier (expected 'MVE1')";
        case CORPUS_ERR_FILE_SIZE_MISMATCH:
            return "File size does not match expected size from header";
        case CORPUS_ERR_DIMENSION_MISMATCH:
            return "Vector dimension mismatch";
        case CORPUS_ERR_NON_FINITE_VALUE:
            return "Vector contains non-finite float value (NaN or Inf)";
        case CORPUS_ERR_ZERO_MAGNITUDE:
            return "Vector has zero or near-zero magnitude (< 1e-6)";
        case CORPUS_ERR_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case CORPUS_ERR_EMPTY_CORPUS:
            return "Corpus contains zero movie records";
        case CORPUS_ERR_INVALID_METADATA:
            return "Movie IDs must be nonzero and unique, and titles must not be empty";
        default:
            return "Unknown corpus error";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* MOVIE_CONTRACT_H */
