#ifndef MOVIE_SEARCH_H
#define MOVIE_SEARCH_H
#include "movie_contract.h"

typedef struct {
    uint64_t id;
    uint32_t index;
    float score;
} MovieResult;

void normalize_movie(const float* raw, float* normalized, uint32_t dim);
void sort_movie_results(MovieResult* results, size_t count);
#endif
