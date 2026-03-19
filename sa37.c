/*
 * Fast SA for length-37 compressed LP pairs.
 * Compile: cc -O3 -march=native -o sa37 sa37.c -lm
 *
 * Runs joint SA on (r_A, r_B) minimizing:
 *   E = sum_{k=1}^{18} |PSD_A(k) + PSD_B(k) - 668|
 *
 * Uses incremental DFT updates: O(18) per move instead of O(37).
 * PRNG: xoshiro256** for speed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define LEN 37
#define N_FREQ 18
#define TARGET 668.0
#define PI 3.14159265358979323846

/* Complex number */
typedef struct { double re, im; } cplx;

static inline cplx cmul(cplx a, cplx b) {
    return (cplx){a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}
static inline cplx cadd(cplx a, cplx b) {
    return (cplx){a.re + b.re, a.im + b.im};
}
static inline double cnorm2(cplx a) {
    return a.re*a.re + a.im*a.im;
}

/* xoshiro256** PRNG */
static uint64_t s[4];

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro_next(void) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

static inline double rand_double(void) {
    return (xoshiro_next() >> 11) * 0x1.0p-53;
}

static inline int rand_int(int n) {
    return (int)(rand_double() * n);
}

static void seed_rng(uint64_t seed) {
    /* splitmix64 to seed xoshiro */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[i] = z ^ (z >> 31);
    }
}

/* Twiddle factors: TWID[k][j] = exp(-2πi(k+1)j/37) */
static cplx TWID[N_FREQ][LEN];

static void init_twiddle(void) {
    for (int k = 0; k < N_FREQ; k++) {
        for (int j = 0; j < LEN; j++) {
            double angle = -2.0 * PI * (k+1) * j / LEN;
            TWID[k][j] = (cplx){cos(angle), sin(angle)};
        }
    }
}

/* Odd values */
static const int ODD_VALS[10] = {-9,-7,-5,-3,-1,1,3,5,7,9};

/* Initialize sequence: odd entries in [-9,9], sum=1, target sum of squares */
static void init_seq(int *seq, int target_sumsq) {
    double avg_sq = (double)target_sumsq / LEN;
    double frac3 = (avg_sq - 1.0) / 8.0;
    if (frac3 < 0) frac3 = 0;
    if (frac3 > 1) frac3 = 1;

    for (int attempt = 0; attempt < 10000; attempt++) {
        for (int i = 0; i < LEN; i++) {
            if (rand_double() < frac3)
                seq[i] = (rand_double() < 0.5) ? -3 : 3;
            else
                seq[i] = (rand_double() < 0.5) ? -1 : 1;
        }
        int sum = 0;
        for (int i = 0; i < LEN; i++) sum += seq[i];

        /* Fix sum to 1 */
        for (int fix = 0; fix < 500; fix++) {
            if (sum == 1) return;
            int idx = rand_int(LEN);
            if (sum > 1 && seq[idx] > -9) {
                seq[idx] -= 2; sum -= 2;
            } else if (sum < 1 && seq[idx] < 9) {
                seq[idx] += 2; sum += 2;
            }
        }
        if (sum == 1) return;
    }
    /* Fallback */
    for (int i = 0; i < LEN; i++) seq[i] = 1;
    seq[0] = -1; seq[1] = -1; /* sum = 37 - 4 = 33... need more */
    int sum = 0;
    for (int i = 0; i < LEN; i++) sum += seq[i];
    for (int i = 0; sum != 1 && i < LEN; i++) {
        if (sum > 1 && seq[i] > -9) { seq[i] -= 2; sum -= 2; }
    }
}

/* Compute DFT at frequencies 1..18 */
static void compute_dft(const int *seq, cplx *dft) {
    for (int k = 0; k < N_FREQ; k++) {
        dft[k] = (cplx){0, 0};
        for (int j = 0; j < LEN; j++) {
            dft[k].re += seq[j] * TWID[k][j].re;
            dft[k].im += seq[j] * TWID[k][j].im;
        }
    }
}

/* Compute energy = sum |PSD_A(k) + PSD_B(k) - 668| */
static double compute_energy(const cplx *dft_A, const cplx *dft_B) {
    double E = 0;
    for (int k = 0; k < N_FREQ; k++) {
        double psd_sum = cnorm2(dft_A[k]) + cnorm2(dft_B[k]);
        E += fabs(psd_sum - TARGET);
    }
    return E;
}

/* Main SA */
static double sa_run(int *rA, int *rB, long long max_iter, double *best_E_out) {
    cplx dft_A[N_FREQ], dft_B[N_FREQ];
    compute_dft(rA, dft_A);
    compute_dft(rB, dft_B);

    double cur_E = compute_energy(dft_A, dft_B);
    double best_E = cur_E;
    int best_A[LEN], best_B[LEN];
    memcpy(best_A, rA, sizeof(int)*LEN);
    memcpy(best_B, rB, sizeof(int)*LEN);

    /* Estimate typical dE for temperature calibration */
    double sum_abs_dE = 0;
    int n_test = 0;
    for (int t = 0; t < 200; t++) {
        int i = rand_int(LEN), j = rand_int(LEN);
        while (j == i) j = rand_int(LEN);
        if (rA[i] == rA[j]) continue;
        double di = rA[j] - rA[i];
        double dj = rA[i] - rA[j];
        cplx new_dft[N_FREQ];
        for (int k = 0; k < N_FREQ; k++) {
            new_dft[k].re = dft_A[k].re + di*TWID[k][i].re + dj*TWID[k][j].re;
            new_dft[k].im = dft_A[k].im + di*TWID[k][i].im + dj*TWID[k][j].im;
        }
        double new_E = 0;
        for (int k = 0; k < N_FREQ; k++)
            new_E += fabs(cnorm2(new_dft[k]) + cnorm2(dft_B[k]) - TARGET);
        sum_abs_dE += fabs(new_E - cur_E);
        n_test++;
    }
    double T_init = (n_test > 0) ? (sum_abs_dE / n_test) * 2.0 : cur_E * 0.1;

    long long accepts = 0;
    long long stale = 0;
    long long phase_len = max_iter / 25;  /* reheat every 4% */

    for (long long it = 0; it < max_iter; it++) {
        /* Reheat when stuck */
        if (stale > phase_len) {
            T_init = best_E * 0.15;
            stale = 0;
            /* Large perturbation: randomize 6 entries of one sequence */
            int *seq = (rand_double() < 0.5) ? rA : rB;
            for (int p = 0; p < 6; p++) {
                int idx = rand_int(LEN);
                seq[idx] = ODD_VALS[rand_int(10)];
            }
            /* Fix sum */
            int sum = 0;
            for (int q = 0; q < LEN; q++) sum += seq[q];
            for (int fix = 0; fix < 200 && sum != 1; fix++) {
                int idx = rand_int(LEN);
                if (sum > 1 && seq[idx] > -9) { seq[idx] -= 2; sum -= 2; }
                else if (sum < 1 && seq[idx] < 9) { seq[idx] += 2; sum += 2; }
            }
            compute_dft(rA, dft_A);
            compute_dft(rB, dft_B);
            cur_E = compute_energy(dft_A, dft_B);
        }

        double progress = (double)it / max_iter;
        double phase_progress = (double)stale / phase_len;
        double T = T_init * pow(1.0 - phase_progress, 1.5);
        if (T < 0.01) T = 0.01;

        /* Choose sequence */
        int *seq;
        cplx *dft, *other_dft;
        if (rand_double() < 0.5) {
            seq = rA; dft = dft_A; other_dft = dft_B;
        } else {
            seq = rB; dft = dft_B; other_dft = dft_A;
        }

        /* Choose move */
        double move_r = rand_double();
        int i = rand_int(LEN), j = rand_int(LEN);
        while (j == i) j = rand_int(LEN);

        double di, dj;
        int new_i, new_j;

        if (move_r < 0.4) {
            /* Swap */
            if (seq[i] == seq[j]) { stale++; continue; }
            di = seq[j] - seq[i];
            dj = seq[i] - seq[j];
            new_i = seq[j];
            new_j = seq[i];
        } else if (move_r < 0.85) {
            /* Change + adjust */
            int old_i = seq[i], old_j = seq[j];
            new_i = ODD_VALS[rand_int(10)];
            new_j = old_i + old_j - new_i;
            if (new_j < -9 || new_j > 9 || (new_j & 1) == 0) { stale++; continue; }
            if (new_i == old_i && new_j == old_j) { stale++; continue; }
            di = new_i - old_i;
            dj = new_j - old_j;
        } else {
            /* 3-entry move: change i,j,k with sum preserved */
            int k2 = rand_int(LEN);
            while (k2 == i || k2 == j) k2 = rand_int(LEN);
            int old_i = seq[i], old_j = seq[j], old_k = seq[k2];
            new_i = ODD_VALS[rand_int(10)];
            new_j = ODD_VALS[rand_int(10)];
            int new_k = old_i + old_j + old_k - new_i - new_j;
            if (new_k < -9 || new_k > 9 || (new_k & 1) == 0) { stale++; continue; }
            /* Compute DFT change for 3 entries */
            double d_i = new_i - old_i, d_j = new_j - old_j, d_k = new_k - old_k;
            cplx new_dft3[N_FREQ];
            double new_E3 = 0;
            for (int kk = 0; kk < N_FREQ; kk++) {
                new_dft3[kk].re = dft[kk].re + d_i*TWID[kk][i].re + d_j*TWID[kk][j].re + d_k*TWID[kk][k2].re;
                new_dft3[kk].im = dft[kk].im + d_i*TWID[kk][i].im + d_j*TWID[kk][j].im + d_k*TWID[kk][k2].im;
                new_E3 += fabs(cnorm2(new_dft3[kk]) + cnorm2(other_dft[kk]) - TARGET);
            }
            double dE3 = new_E3 - cur_E;
            if (dE3 < 0 || rand_double() < exp(-dE3 / T)) {
                seq[i] = new_i; seq[j] = new_j; seq[k2] = new_k;
                memcpy(dft, new_dft3, sizeof(cplx)*N_FREQ);
                cur_E = new_E3;
                accepts++;
                if (cur_E < best_E) {
                    best_E = cur_E;
                    memcpy(best_A, rA, sizeof(int)*LEN);
                    memcpy(best_B, rB, sizeof(int)*LEN);
                    stale = 0;
                    if (best_E < 0.5) {
                        memcpy(rA, best_A, sizeof(int)*LEN);
                        memcpy(rB, best_B, sizeof(int)*LEN);
                        *best_E_out = best_E;
                        return best_E;
                    }
                } else stale++;
            } else stale++;
            continue; /* skip the 2-entry accept below */
        }

        /* Incremental DFT update and energy computation (2-entry moves) */
        double new_E = 0;
        cplx new_dft[N_FREQ];
        for (int k = 0; k < N_FREQ; k++) {
            new_dft[k].re = dft[k].re + di*TWID[k][i].re + dj*TWID[k][j].re;
            new_dft[k].im = dft[k].im + di*TWID[k][i].im + dj*TWID[k][j].im;
            new_E += fabs(cnorm2(new_dft[k]) + cnorm2(other_dft[k]) - TARGET);
        }

        double dE = new_E - cur_E;
        if (dE < 0 || rand_double() < exp(-dE / T)) {
            seq[i] = new_i;
            seq[j] = new_j;
            memcpy(dft, new_dft, sizeof(cplx)*N_FREQ);
            cur_E = new_E;
            accepts++;

            if (cur_E < best_E) {
                best_E = cur_E;
                memcpy(best_A, rA, sizeof(int)*LEN);
                memcpy(best_B, rB, sizeof(int)*LEN);
                stale = 0;

                if (best_E < 0.5) {
                    memcpy(rA, best_A, sizeof(int)*LEN);
                    memcpy(rB, best_B, sizeof(int)*LEN);
                    *best_E_out = best_E;
                    return best_E;
                }
            } else stale++;
        } else stale++;

        if ((it+1) % 50000000 == 0) {
            fprintf(stderr, "  %lld M: E=%.1f best=%.1f T=%.1f acc=%.1f%%\n",
                    (it+1)/1000000, cur_E, best_E, T,
                    100.0*accepts/(it+1));
        }
    }

    memcpy(rA, best_A, sizeof(int)*LEN);
    memcpy(rB, best_B, sizeof(int)*LEN);
    *best_E_out = best_E;
    return best_E;
}

int main(int argc, char **argv) {
    long long max_iter = 500000000LL; /* 500M default */
    int n_trials = 10;
    uint64_t base_seed = 42;

    if (argc > 1) n_trials = atoi(argv[1]);
    if (argc > 2) max_iter = atoll(argv[2]);
    if (argc > 3) base_seed = atoll(argv[3]);

    init_twiddle();

    fprintf(stderr, "SA37: %d trials x %lld iterations\n", n_trials, max_iter);

    double global_best = 1e18;
    int global_best_A[LEN], global_best_B[LEN];

    for (int trial = 0; trial < n_trials; trial++) {
        seed_rng(base_seed + trial * 104729);

        int rA[LEN], rB[LEN];
        int sumsq_A = 200 + rand_int(250);
        int sumsq_B = 650 - sumsq_A;
        init_seq(rA, sumsq_A);
        init_seq(rB, sumsq_B);

        double best_E;
        clock_t t0 = clock();
        sa_run(rA, rB, max_iter, &best_E);
        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;

        fflush(stdout); fprintf(stderr, "Trial %d: E=%.1f  (%.1fs)\n", trial, best_E, dt);

        if (best_E < global_best) {
            global_best = best_E;
            memcpy(global_best_A, rA, sizeof(int)*LEN);
            memcpy(global_best_B, rB, sizeof(int)*LEN);
        }

        if (best_E < 0.5) {
            fprintf(stderr, "\n*** EXACT SOLUTION FOUND! ***\n");
            printf("A = [");
            for (int i = 0; i < LEN; i++) printf("%d%s", global_best_A[i], i<LEN-1?",":"");
            printf("]\n");
            printf("B = [");
            for (int i = 0; i < LEN; i++) printf("%d%s", global_best_B[i], i<LEN-1?",":"");
            printf("]\n");
            return 0;
        }
    }

    fprintf(stderr, "\nBest energy: %.1f\n", global_best);
    printf("BEST_A = [");
    for (int i = 0; i < LEN; i++) printf("%d%s", global_best_A[i], i<LEN-1?",":"");
    printf("]\n");
    printf("BEST_B = [");
    for (int i = 0; i < LEN; i++) printf("%d%s", global_best_B[i], i<LEN-1?",":"");
    printf("]\n");
    printf("ENERGY = %.6f\n", global_best);

    return 0;
}
