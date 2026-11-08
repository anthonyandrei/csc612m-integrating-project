/** @file
 * @brief Checks partitioned top-K selection against the full-sort reference.
 */
#include "movie_search.h"
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static uint32_t rng_state = 12345;
static uint32_t next_rand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

/* Few distinct scores force many exact ties; IDs are unique but shuffled. */
static void make_candidates(MovieResult* c, size_t n, uint32_t distinct) {
    for (size_t i = 0; i < n; i++) {
        c[i].id = ((uint64_t)i * 2654435761ull) % 1000003ull + 1;
        c[i].index = (uint32_t)i;
        c[i].score = (float)(next_rand() % distinct) / 4.0f;
    }
}

static void check_case(const MovieResult* c, size_t n, const CandidateRange* parts,
                       size_t np, size_t k) {
    MovieResult* scratch = malloc((np * k + 1) * sizeof(*scratch));
    MovieResult* out = malloc((k + 1) * sizeof(*out));
    MovieResult* work = malloc((n + 1) * sizeof(*work));
    MovieResult* ref = malloc((k + 1) * sizeof(*ref));
    CHECK(scratch && out && work && ref);
    if (scratch && out && work && ref) {
        CHECK(partitions_are_valid(n, parts, np));
        size_t got = select_top_k_partitioned(c, n, parts, np, k, scratch, out);
        size_t want = select_top_k_reference(c, n, k, work, ref);
        CHECK(want == (n < k ? n : k));
        CHECK(got == want);
        for (size_t i = 0; i < want && i < got; i++)
            CHECK(out[i].id == ref[i].id && out[i].score == ref[i].score);
    }
    free(scratch);
    free(out);
    free(work);
    free(ref);
}


/* Prints one small worked example so the selection structure can be seen. */
static void print_walkthrough(void) {
    enum { N = 12, K = 3, P = 4 };
    const float scores[N] = {0.90f, 0.40f, 0.90f, 0.70f, 0.10f, 0.90f,
                             0.70f, 0.30f, 0.80f, 0.90f, 0.20f, 0.70f};
    const uint64_t ids[N] = {105, 101, 103, 107, 102, 104, 106, 110, 109, 108, 111, 112};
    const CandidateRange parts[P] = {{0, 3}, {3, 3}, {3, 8}, {8, 12}}; /* uneven + one empty */
    MovieResult c[N], local[P][K], scratch[P * K], out[K], work[N], ref[K];
    size_t local_n[P], got, want;

    for (size_t i = 0; i < N; i++) {
        c[i].id = ids[i];
        c[i].index = (uint32_t)i;
        c[i].score = scores[i];
    }
    printf("\nWorked example: N=%d candidates, K=%d, %d partitions (uneven, one empty)\n", N, K, P);
    printf("Ranking: score descending, then ID ascending for exact ties\n");
    printf("--------------------------------------------------------------\n");
    for (size_t p = 0; p < P; p++) {
        CandidateRange whole = {0, parts[p].end - parts[p].begin};
        local_n[p] = select_top_k_partitioned(c + parts[p].begin, whole.end, &whole, 1, K, scratch, local[p]);
        printf("P%zu [%2zu,%2zu) %zu candidates  ->  local top-%d:", p, parts[p].begin, parts[p].end,
               parts[p].end - parts[p].begin, K);
        if (local_n[p] == 0)
            printf(" (empty)");
        for (size_t i = 0; i < local_n[p]; i++)
            printf("  %llu(%.2f)", (unsigned long long)local[p][i].id, local[p][i].score);
        printf("\n");
    }
    got = select_top_k_partitioned(c, N, parts, P, K, scratch, out);
    want = select_top_k_reference(c, N, K, work, ref);
    printf("--------------------------------------------------------------\n");
    printf("merged + sorted   :");
    for (size_t i = 0; i < got; i++)
        printf("  %llu(%.2f)", (unsigned long long)out[i].id, out[i].score);
    printf("\nfull-sort reference:");
    for (size_t i = 0; i < want; i++)
        printf("  %llu(%.2f)", (unsigned long long)ref[i].id, ref[i].score);
    printf("\n%s\n\n", got == want ? "match: identical ordered IDs" : "MISMATCH");
    CHECK(got == want);
    for (size_t i = 0; i < want && i < got; i++)
        CHECK(out[i].id == ref[i].id);
}

int main(void) {
    print_walkthrough();

    const size_t sizes[] = {0, 1, 5, 37, 200};
    const size_t ks[] = {1, 2, 3, 10, 40, 250};
    const size_t part_counts[] = {1, 2, 3, 7, 50};
    const uint32_t distinct[] = {1, 3, 1000}; /* all-tied, heavy ties, near-unique */
    CandidateRange parts[64];

    for (size_t s = 0; s < 5; s++) {
        for (size_t d = 0; d < 3; d++) {
            MovieResult* c = malloc((sizes[s] + 1) * sizeof(*c));
            CHECK(c != NULL);
            if (!c)
                continue;
            make_candidates(c, sizes[s], distinct[d]);
            for (size_t k = 0; k < 6; k++) {        /* includes K > candidate count */
                for (size_t p = 0; p < 5; p++) {    /* includes more partitions than candidates */
                    size_t np = make_even_partitions(sizes[s], part_counts[p], parts);
                    check_case(c, sizes[s], parts, np, ks[k]);
                }
            }
            free(c);
        }
    }

    /* Explicit uneven layout with empty partitions at the start, middle and end. */
    {
        MovieResult c[20];
        const CandidateRange uneven[] = {{0, 0}, {0, 7}, {7, 7}, {7, 8}, {8, 20}, {20, 20}};
        make_candidates(c, 20, 3);
        for (size_t k = 1; k <= 25; k++)
            check_case(c, 20, uneven, 6, k);
    }

    /* Coverage: even partitions are always valid and sizes differ by at most one. */
    for (size_t n = 0; n <= 100; n++) {
        for (size_t p = 1; p <= 64; p++) {
            size_t np = make_even_partitions(n, p, parts), mn = n, mx = 0;
            CHECK(np == p && partitions_are_valid(n, parts, np));
            for (size_t i = 0; i < np; i++) {
                size_t size = parts[i].end - parts[i].begin;
                if (size < mn)
                    mn = size;
                if (size > mx)
                    mx = size;
            }
            CHECK(mx - mn <= 1);
        }
    }

    /* Invalid layouts: gap, overlap, short coverage, overrun, reversed range. */
    {
        const CandidateRange gap[] = {{0, 3}, {4, 10}};
        const CandidateRange overlap[] = {{0, 6}, {5, 10}};
        const CandidateRange shortc[] = {{0, 4}, {4, 9}};
        const CandidateRange overrun[] = {{0, 5}, {5, 11}};
        const CandidateRange reversed[] = {{0, 5}, {5, 3}, {3, 10}};
        CHECK(!partitions_are_valid(10, gap, 2));
        CHECK(!partitions_are_valid(10, overlap, 2));
        CHECK(!partitions_are_valid(10, shortc, 2));
        CHECK(!partitions_are_valid(10, overrun, 2));
        CHECK(!partitions_are_valid(10, reversed, 3));
        CHECK(!partitions_are_valid(10, NULL, 0));
    }

    /* Query exclusion: build candidates from 10 movies, skipping the query at index 4. */
    {
        MovieResult all[10], c[9], out[9], scratch[27];
        CandidateRange p[3];
        size_t n = 0;
        make_candidates(all, 10, 3);
        for (uint32_t i = 0; i < 10; i++) {
            if (i != 4)
                c[n++] = all[i];
        }
        make_even_partitions(n, 3, p);
        size_t got = select_top_k_partitioned(c, n, p, 3, 9, scratch, out);
        CHECK(got == 9);
        for (size_t i = 0; i < got; i++)
            CHECK(out[i].id != all[4].id);
    }

    if (failures) {
        fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    puts("Partitioned top-K checks passed.");
    return 0;
}