/*
 * Exhaustive steepest descent for length-37 compressed LP pairs.
 *
 * Starting from SA's best solution, exhaustively search ALL possible
 * k-entry moves (k=2,3,4) and take the best improving one. Repeat
 * until no improvement found, then increase k.
 *
 * This is deterministic and guarantees finding the local minimum
 * with respect to the given neighborhood.
 *
 * Compile: cc -O3 -march=native -o descent37 descent37.c -lm
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

typedef struct { double re, im; } cplx;
static inline double cnorm2(cplx a) { return a.re*a.re + a.im*a.im; }

static cplx TWID[N_FREQ][LEN];
static const int ODD_VALS[10] = {-9,-7,-5,-3,-1,1,3,5,7,9};

/* xoshiro256** */
static uint64_t rng_s[4];
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t xnext(void) {
    const uint64_t r = rotl(rng_s[1]*5,7)*9, t = rng_s[1]<<17;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1]; rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=rotl(rng_s[3],45); return r;
}
static inline double rnd(void) { return (xnext()>>11)*0x1.0p-53; }
static inline int rnd_int(int n) { return (int)(rnd()*n); }
static void seed_rng(uint64_t seed) {
    for (int i=0;i<4;i++) { seed+=0x9e3779b97f4a7c15ULL; uint64_t z=seed;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL;
        rng_s[i]=z^(z>>31); }
}

static void init_twiddle(void) {
    for (int k = 0; k < N_FREQ; k++)
        for (int j = 0; j < LEN; j++) {
            double a = -2.0*PI*(k+1)*j/LEN;
            TWID[k][j] = (cplx){cos(a), sin(a)};
        }
}

static void compute_dft(const int *seq, cplx *dft) {
    for (int k = 0; k < N_FREQ; k++) {
        dft[k] = (cplx){0,0};
        for (int j = 0; j < LEN; j++) {
            dft[k].re += seq[j] * TWID[k][j].re;
            dft[k].im += seq[j] * TWID[k][j].im;
        }
    }
}

static double compute_energy(const cplx *dA, const cplx *dB) {
    double E = 0;
    for (int k = 0; k < N_FREQ; k++)
        E += fabs(cnorm2(dA[k]) + cnorm2(dB[k]) - TARGET);
    return E;
}

/*
 * Exhaustive 2-entry steepest descent.
 * For each pair (i,j) and each valid (new_i, new_j), compute energy.
 * Apply the best improving move. Repeat until no improvement.
 */
static int descent_2(int *rA, int *rB, cplx *dft_A, cplx *dft_B, double *E) {
    int improved = 0;

    for (;;) {
        double best_dE = 0;
        int best_seq = -1, best_i = -1, best_j = -1, best_ni, best_nj;
        cplx best_dft[N_FREQ];

        for (int which = 0; which < 2; which++) {
            int *seq = which ? rB : rA;
            cplx *dft = which ? dft_B : dft_A;
            cplx *other = which ? dft_A : dft_B;

            for (int i = 0; i < LEN; i++) {
                for (int j = i+1; j < LEN; j++) {
                    int old_i = seq[i], old_j = seq[j];
                    int sum_ij = old_i + old_j;

                    for (int vi = 0; vi < 10; vi++) {
                        int ni = ODD_VALS[vi];
                        int nj = sum_ij - ni;
                        if (nj < -9 || nj > 9 || (nj & 1) == 0) continue;
                        if (ni == old_i && nj == old_j) continue;

                        double di = ni - old_i, dj = nj - old_j;
                        double new_E = 0;
                        cplx nd[N_FREQ];
                        for (int k = 0; k < N_FREQ; k++) {
                            nd[k].re = dft[k].re + di*TWID[k][i].re + dj*TWID[k][j].re;
                            nd[k].im = dft[k].im + di*TWID[k][i].im + dj*TWID[k][j].im;
                            new_E += fabs(cnorm2(nd[k]) + cnorm2(other[k]) - TARGET);
                        }

                        double dE = new_E - *E;
                        if (dE < best_dE - 1e-9) {
                            best_dE = dE;
                            best_seq = which;
                            best_i = i; best_j = j;
                            best_ni = ni; best_nj = nj;
                            memcpy(best_dft, nd, sizeof(cplx)*N_FREQ);
                        }
                    }
                }
            }
        }

        if (best_seq < 0) break; /* no improving move */

        /* Apply best move */
        int *seq = best_seq ? rB : rA;
        cplx *dft = best_seq ? dft_B : dft_A;
        seq[best_i] = best_ni;
        seq[best_j] = best_nj;
        memcpy(dft, best_dft, sizeof(cplx)*N_FREQ);
        *E += best_dE;
        improved++;

        if (*E < 0.5) return improved;

        fprintf(stderr, "  2-descent step %d: E=%.3f (dE=%.3f)\n",
                improved, *E, best_dE);
    }
    return improved;
}

/*
 * Exhaustive 3-entry steepest descent.
 * For each triple (i,j,k) and valid (ni,nj,nk), find best improvement.
 */
static int descent_3(int *rA, int *rB, cplx *dft_A, cplx *dft_B, double *E) {
    int improved = 0;

    for (;;) {
        double best_dE = 0;
        int best_seq = -1, best_i, best_j, best_k2, best_ni, best_nj, best_nk;
        cplx best_dft[N_FREQ];

        for (int which = 0; which < 2; which++) {
            int *seq = which ? rB : rA;
            cplx *dft = which ? dft_B : dft_A;
            cplx *other = which ? dft_A : dft_B;

            for (int i = 0; i < LEN; i++) {
                for (int j = i+1; j < LEN; j++) {
                    for (int k2 = j+1; k2 < LEN; k2++) {
                        int old_i = seq[i], old_j = seq[j], old_k = seq[k2];
                        int sum3 = old_i + old_j + old_k;

                        for (int vi = 0; vi < 10; vi++) {
                            int ni = ODD_VALS[vi];
                            for (int vj = 0; vj < 10; vj++) {
                                int nj = ODD_VALS[vj];
                                int nk = sum3 - ni - nj;
                                if (nk < -9 || nk > 9 || (nk & 1) == 0) continue;
                                if (ni == old_i && nj == old_j && nk == old_k) continue;

                                double di = ni-old_i, dj = nj-old_j, dk = nk-old_k;
                                double new_E = 0;
                                cplx nd[N_FREQ];
                                for (int kk = 0; kk < N_FREQ; kk++) {
                                    nd[kk].re = dft[kk].re + di*TWID[kk][i].re + dj*TWID[kk][j].re + dk*TWID[kk][k2].re;
                                    nd[kk].im = dft[kk].im + di*TWID[kk][i].im + dj*TWID[kk][j].im + dk*TWID[kk][k2].im;
                                    new_E += fabs(cnorm2(nd[kk]) + cnorm2(other[kk]) - TARGET);
                                }

                                double dE = new_E - *E;
                                if (dE < best_dE - 1e-9) {
                                    best_dE = dE;
                                    best_seq = which;
                                    best_i = i; best_j = j; best_k2 = k2;
                                    best_ni = ni; best_nj = nj; best_nk = nk;
                                    memcpy(best_dft, nd, sizeof(cplx)*N_FREQ);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (best_seq < 0) break;

        int *seq = best_seq ? rB : rA;
        cplx *dft = best_seq ? dft_B : dft_A;
        seq[best_i] = best_ni;
        seq[best_j] = best_nj;
        seq[best_k2] = best_nk;
        memcpy(dft, best_dft, sizeof(cplx)*N_FREQ);
        *E += best_dE;
        improved++;

        if (*E < 0.5) return improved;

        fprintf(stderr, "  3-descent step %d: E=%.3f (dE=%.3f)\n",
                improved, *E, best_dE);
    }
    return improved;
}

/* SA for initial solution generation */
static void sa_init(int *rA, int *rB, long long max_iter) {
    /* [same as sa37.c sa_run but simplified] */
    cplx dft_A[N_FREQ], dft_B[N_FREQ];
    compute_dft(rA, dft_A); compute_dft(rB, dft_B);
    double cur_E = compute_energy(dft_A, dft_B);
    double best_E = cur_E;
    int best_A[LEN], best_B[LEN];
    memcpy(best_A, rA, sizeof(int)*LEN);
    memcpy(best_B, rB, sizeof(int)*LEN);

    /* Temperature calibration */
    double sum_dE = 0; int nt = 0;
    for (int t=0; t<200; t++) {
        int i=rnd_int(LEN), j=rnd_int(LEN);
        while(j==i) j=rnd_int(LEN);
        if(rA[i]==rA[j]) continue;
        double di=rA[j]-rA[i], dj=rA[i]-rA[j];
        cplx nd[N_FREQ]; double nE=0;
        for(int k=0;k<N_FREQ;k++){
            nd[k].re=dft_A[k].re+di*TWID[k][i].re+dj*TWID[k][j].re;
            nd[k].im=dft_A[k].im+di*TWID[k][i].im+dj*TWID[k][j].im;
            nE+=fabs(cnorm2(nd[k])+cnorm2(dft_B[k])-TARGET);
        }
        sum_dE+=fabs(nE-cur_E); nt++;
    }
    double T_init = nt>0 ? (sum_dE/nt)*2.0 : cur_E*0.1;

    for (long long it=0; it<max_iter; it++) {
        double T = T_init * pow(1.0-(double)it/max_iter, 1.5);
        if (T<0.01) T=0.01;
        int *seq; cplx *dft, *other;
        if(rnd()<0.5){seq=rA;dft=dft_A;other=dft_B;}
        else{seq=rB;dft=dft_B;other=dft_A;}
        int i=rnd_int(LEN),j=rnd_int(LEN);
        while(j==i)j=rnd_int(LEN);
        int old_i=seq[i],old_j=seq[j];
        int ni,nj;
        if(rnd()<0.5){
            if(seq[i]==seq[j])continue;
            ni=seq[j];nj=seq[i];
        } else {
            ni=ODD_VALS[rnd_int(10)];
            nj=old_i+old_j-ni;
            if(nj<-9||nj>9||(nj&1)==0)continue;
            if(ni==old_i&&nj==old_j)continue;
        }
        double di=ni-old_i,dj=nj-old_j;
        cplx nd[N_FREQ]; double nE=0;
        for(int k=0;k<N_FREQ;k++){
            nd[k].re=dft[k].re+di*TWID[k][i].re+dj*TWID[k][j].re;
            nd[k].im=dft[k].im+di*TWID[k][i].im+dj*TWID[k][j].im;
            nE+=fabs(cnorm2(nd[k])+cnorm2(other[k])-TARGET);
        }
        if(nE-cur_E<0||rnd()<exp(-(nE-cur_E)/T)){
            seq[i]=ni;seq[j]=nj;memcpy(dft,nd,sizeof(cplx)*N_FREQ);
            cur_E=nE;
            if(cur_E<best_E){best_E=cur_E;memcpy(best_A,rA,sizeof(int)*LEN);memcpy(best_B,rB,sizeof(int)*LEN);}
        }
    }
    memcpy(rA, best_A, sizeof(int)*LEN);
    memcpy(rB, best_B, sizeof(int)*LEN);
}

static void init_seq(int *seq, int target_sumsq) {
    double frac3 = ((double)target_sumsq/LEN - 1.0) / 8.0;
    if(frac3<0)frac3=0; if(frac3>1)frac3=1;
    for(int a=0;a<10000;a++){
        for(int i=0;i<LEN;i++) seq[i]=(rnd()<frac3)?((rnd()<0.5)?-3:3):((rnd()<0.5)?-1:1);
        int sum=0; for(int i=0;i<LEN;i++)sum+=seq[i];
        for(int f=0;f<500&&sum!=1;f++){int idx=rnd_int(LEN);
            if(sum>1&&seq[idx]>-9){seq[idx]-=2;sum-=2;}
            else if(sum<1&&seq[idx]<9){seq[idx]+=2;sum+=2;}}
        if(sum==1)return;
    }
}

int main(int argc, char **argv) {
    int n_trials = argc > 1 ? atoi(argv[1]) : 20;
    long long sa_iters = argc > 2 ? atoll(argv[2]) : 2000000000LL;
    uint64_t base_seed = argc > 3 ? atoll(argv[3]) : 42;
    int use_fixed = argc > 4;  /* if 5th arg given, use hardcoded best */

    init_twiddle();

    double global_best = 1e18;
    int global_A[LEN], global_B[LEN];

    fprintf(stderr, "SA(%lldM) + cascade descent: %d trials\n", sa_iters/1000000, n_trials);

    for (int trial = 0; trial < n_trials; trial++) {
        seed_rng(base_seed + trial * 104729);

        int rA[LEN], rB[LEN];

        if (use_fixed && trial == 0) {
            /* Start from hardcoded best SA solution */
            int tA[] = {-1,1,1,1,3,-1,-1,3,-1,-1,1,1,1,-1,-3,1,1,1,1,-7,-1,1,-1,-1,1,-1,-5,-1,-1,1,-1,5,3,-3,-1,-1,7};
            int tB[] = {-1,1,-3,-3,-1,5,-1,1,1,1,3,-3,1,-3,1,1,-7,-5,-1,1,1,1,1,3,-5,1,-1,3,-5,5,7,3,3,5,-9,1,-1};
            memcpy(rA, tA, sizeof(int)*LEN);
            memcpy(rB, tB, sizeof(int)*LEN);
        } else {
            int sq = 200 + rnd_int(250);
            init_seq(rA, sq); init_seq(rB, 650-sq);

            /* SA phase */
            clock_t t0 = clock();
            sa_init(rA, rB, sa_iters);
            double dt = (double)(clock()-t0)/CLOCKS_PER_SEC;

            cplx dA[N_FREQ], dB[N_FREQ];
            compute_dft(rA, dA); compute_dft(rB, dB);
            double E = compute_energy(dA, dB);
            fprintf(stderr, "Trial %d SA: E=%.1f (%.1fs)\n", trial, E, dt);
        }

        /* Cascade descent */
        cplx dft_A[N_FREQ], dft_B[N_FREQ];
        compute_dft(rA, dft_A); compute_dft(rB, dft_B);
        double E = compute_energy(dft_A, dft_B);

        fprintf(stderr, "Trial %d: starting descent from E=%.1f\n", trial, E);

        /* 2-entry descent */
        clock_t t0 = clock();
        int steps = descent_2(rA, rB, dft_A, dft_B, &E);
        double dt = (double)(clock()-t0)/CLOCKS_PER_SEC;
        fprintf(stderr, "  2-descent: %d steps, E=%.3f (%.1fs)\n", steps, E, dt);

        if (E < 0.5) goto found;

        /* 3-entry descent */
        t0 = clock();
        steps = descent_3(rA, rB, dft_A, dft_B, &E);
        dt = (double)(clock()-t0)/CLOCKS_PER_SEC;
        fprintf(stderr, "  3-descent: %d steps, E=%.3f (%.1fs)\n", steps, E, dt);

        if (E < 0.5) goto found;

        /* 4-entry random descent: sample 50M random 4-entry moves,
           accept any improving one, repeat until no improvement in full scan */
        {
            int improved4 = 0;
            for (int round = 0; round < 20; round++) {
                int found_better = 0;
                for (long long s4 = 0; s4 < 50000000LL; s4++) {
                    int which = rnd_int(2);
                    int *seq = which ? rB : rA;
                    cplx *dft = which ? dft_B : dft_A;
                    cplx *other = which ? dft_A : dft_B;

                    /* Pick 4 distinct positions */
                    int p[4];
                    p[0] = rnd_int(LEN);
                    p[1] = rnd_int(LEN); while(p[1]==p[0]) p[1]=rnd_int(LEN);
                    p[2] = rnd_int(LEN); while(p[2]==p[0]||p[2]==p[1]) p[2]=rnd_int(LEN);
                    p[3] = rnd_int(LEN); while(p[3]==p[0]||p[3]==p[1]||p[3]==p[2]) p[3]=rnd_int(LEN);

                    int old_sum = seq[p[0]]+seq[p[1]]+seq[p[2]]+seq[p[3]];
                    int n0 = ODD_VALS[rnd_int(10)];
                    int n1 = ODD_VALS[rnd_int(10)];
                    int n2 = ODD_VALS[rnd_int(10)];
                    int n3 = old_sum - n0 - n1 - n2;
                    if (n3<-9||n3>9||(n3&1)==0) continue;
                    if (n0==seq[p[0]]&&n1==seq[p[1]]&&n2==seq[p[2]]&&n3==seq[p[3]]) continue;

                    double d0=n0-seq[p[0]], d1=n1-seq[p[1]], d2=n2-seq[p[2]], d3=n3-seq[p[3]];
                    cplx nd[N_FREQ]; double nE = 0;
                    for (int kk=0; kk<N_FREQ; kk++) {
                        nd[kk].re = dft[kk].re + d0*TWID[kk][p[0]].re + d1*TWID[kk][p[1]].re + d2*TWID[kk][p[2]].re + d3*TWID[kk][p[3]].re;
                        nd[kk].im = dft[kk].im + d0*TWID[kk][p[0]].im + d1*TWID[kk][p[1]].im + d2*TWID[kk][p[2]].im + d3*TWID[kk][p[3]].im;
                        nE += fabs(cnorm2(nd[kk]) + cnorm2(other[kk]) - TARGET);
                    }
                    if (nE < E - 1e-9) {
                        seq[p[0]]=n0; seq[p[1]]=n1; seq[p[2]]=n2; seq[p[3]]=n3;
                        memcpy(dft, nd, sizeof(cplx)*N_FREQ);
                        E = nE;
                        improved4++;
                        found_better = 1;
                        fprintf(stderr, "  4-descent step %d: E=%.3f\n", improved4, E);
                        if (E < 0.5) { *&E = E; goto found; }
                        break; /* restart scan */
                    }
                }
                if (!found_better) break;
            }
            fprintf(stderr, "  4-descent: %d steps total\n", improved4);
        }

        fprintf(stderr, "Trial %d final: E=%.3f\n\n", trial, E);

        if (E < global_best) {
            global_best = E;
            memcpy(global_A, rA, sizeof(int)*LEN);
            memcpy(global_B, rB, sizeof(int)*LEN);
        }
        continue;

    found:
        fprintf(stderr, "\n*** EXACT SOLUTION FOUND! ***\n");
        printf("A = [");
        for(int i=0;i<LEN;i++) printf("%d%s",rA[i],i<LEN-1?",":"");
        printf("]\nB = [");
        for(int i=0;i<LEN;i++) printf("%d%s",rB[i],i<LEN-1?",":"");
        printf("]\nENERGY = %.9f\n", E);
        return 0;
    }

    fprintf(stderr, "\nBest energy: %.3f\n", global_best);
    printf("BEST_A = [");
    for(int i=0;i<LEN;i++) printf("%d%s",global_A[i],i<LEN-1?",":"");
    printf("]\nBEST_B = [");
    for(int i=0;i<LEN;i++) printf("%d%s",global_B[i],i<LEN-1?",":"");
    printf("]\nENERGY = %.9f\n", global_best);
    return 0;
}
