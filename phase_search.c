/*
 * Phase-space search for length-37 compressed LP pairs.
 *
 * Instead of searching over integer sequences, search over DFT phases.
 *
 * Parameterization: at each frequency k = 1..18, we have
 *   PSD_A(k) = t_k, PSD_B(k) = 668 - t_k  (LP constraint)
 *   DFT_A(k) = sqrt(t_k) * e^{i*phi_k}
 *   DFT_B(k) = sqrt(668-t_k) * e^{i*psi_k}
 *
 * Variables: t_k (18 reals), phi_k (18 angles), psi_k (18 angles) = 54 params
 *
 * Constraint: IDFT gives odd integers in [-9,9] with sum = 1 for both A,B
 *
 * Energy: sum of squared distances from each IDFT entry to nearest valid odd int
 *
 * This is a CONTINUOUS optimization over (t, phi, psi), with rounding
 * at the end. The LP PSD constraint is EXACTLY satisfied by construction.
 *
 * Compile: cc -O3 -march=native -o phase_search phase_search.c -lm
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

/* xoshiro256** */
static uint64_t rng_s[4];
static inline uint64_t rotl(const uint64_t x, int k){return(x<<k)|(x>>(64-k));}
static inline uint64_t xnext(void){
    const uint64_t r=rotl(rng_s[1]*5,7)*9,t=rng_s[1]<<17;
    rng_s[2]^=rng_s[0];rng_s[3]^=rng_s[1];rng_s[1]^=rng_s[2];rng_s[0]^=rng_s[3];
    rng_s[2]^=t;rng_s[3]=rotl(rng_s[3],45);return r;}
static inline double rnd(void){return(xnext()>>11)*0x1.0p-53;}
static void seed_rng(uint64_t s){
    for(int i=0;i<4;i++){s+=0x9e3779b97f4a7c15ULL;uint64_t z=s;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;
    rng_s[i]=z^(z>>31);}}

/* Reconstruct real sequence from DFT magnitudes and phases.
 * F(0) = 1 (sum=1), F(k) = mag*e^{i*phase}, F(37-k) = conj(F(k))
 * x[j] = (1/37) * (F(0) + 2*sum_{k=1}^{18} Re(F(k)*exp(2*pi*i*k*j/37)))
 */
static void reconstruct(const double *mag, const double *phase, double *x) {
    for (int j = 0; j < LEN; j++) {
        double val = 1.0; /* F(0) = 1 */
        for (int k = 0; k < N_FREQ; k++) {
            double angle = phase[k] + 2.0*PI*(k+1)*j/LEN;
            val += 2.0 * mag[k] * cos(angle);
        }
        x[j] = val / LEN;
    }
}

/* Distance from x to nearest valid odd integer in [-9,9] */
static double nearest_odd_dist(double x) {
    /* Nearest odd integer */
    double r = round((x-1)/2)*2 + 1;
    if (r < -9) r = -9;
    if (r > 9) r = 9;
    /* Ensure odd */
    double d = x - r;
    return d*d;
}

static int nearest_odd(double x) {
    int r = (int)round((x-1.0)/2.0)*2 + 1;
    if (r < -9) r = -9;
    if (r > 9) r = 9;
    if ((r & 1) == 0) r += (x > r) ? 1 : -1;
    if (r < -9) r = -9;
    if (r > 9) r = 9;
    return r;
}

/* Energy: sum of squared distances to nearest valid integers for both A,B.
 * Also penalize sum != 1 */
static double energy(const double *t, const double *phi, const double *psi) {
    double mag_A[N_FREQ], mag_B[N_FREQ];
    for (int k = 0; k < N_FREQ; k++) {
        double tk = t[k];
        if (tk < 0) tk = 0;
        if (tk > TARGET) tk = TARGET;
        mag_A[k] = sqrt(tk);
        mag_B[k] = sqrt(TARGET - tk);
    }

    double xA[LEN], xB[LEN];
    reconstruct(mag_A, phi, xA);
    reconstruct(mag_B, psi, xB);

    double E = 0;
    double sum_A = 0, sum_B = 0;
    for (int j = 0; j < LEN; j++) {
        E += nearest_odd_dist(xA[j]);
        E += nearest_odd_dist(xB[j]);
        sum_A += xA[j];
        sum_B += xB[j];
    }
    /* Penalize sum != 1 (should be ~1 by construction since F(0)=1) */
    E += 100.0 * ((sum_A-1)*(sum_A-1) + (sum_B-1)*(sum_B-1));
    return E;
}

/* After optimization, round to integers and check LP PSD */
static double round_and_check(const double *t, const double *phi, const double *psi,
                              int *rA, int *rB) {
    double mag_A[N_FREQ], mag_B[N_FREQ];
    for (int k = 0; k < N_FREQ; k++) {
        double tk = t[k]; if(tk<0)tk=0; if(tk>TARGET)tk=TARGET;
        mag_A[k] = sqrt(tk);
        mag_B[k] = sqrt(TARGET - tk);
    }
    double xA[LEN], xB[LEN];
    reconstruct(mag_A, phi, xA);
    reconstruct(mag_B, psi, xB);

    for (int j = 0; j < LEN; j++) {
        rA[j] = nearest_odd(xA[j]);
        rB[j] = nearest_odd(xB[j]);
    }

    /* Fix sums */
    int sA=0, sB=0;
    for(int j=0;j<LEN;j++){sA+=rA[j];sB+=rB[j];}
    for(int f=0;f<200&&sA!=1;f++){int i=(int)(rnd()*LEN);
        if(sA>1&&rA[i]>-9){rA[i]-=2;sA-=2;}
        else if(sA<1&&rA[i]<9){rA[i]+=2;sA+=2;}}
    for(int f=0;f<200&&sB!=1;f++){int i=(int)(rnd()*LEN);
        if(sB>1&&rB[i]>-9){rB[i]-=2;sB-=2;}
        else if(sB<1&&rB[i]<9){rB[i]+=2;sB+=2;}}

    /* Compute actual LP energy */
    double dft_A_re[N_FREQ], dft_A_im[N_FREQ], dft_B_re[N_FREQ], dft_B_im[N_FREQ];
    for (int k = 0; k < N_FREQ; k++) {
        dft_A_re[k]=dft_A_im[k]=dft_B_re[k]=dft_B_im[k]=0;
        for (int j = 0; j < LEN; j++) {
            double a = -2.0*PI*(k+1)*j/LEN;
            dft_A_re[k] += rA[j]*cos(a); dft_A_im[k] += rA[j]*sin(a);
            dft_B_re[k] += rB[j]*cos(a); dft_B_im[k] += rB[j]*sin(a);
        }
    }
    double lp_E = 0;
    for (int k = 0; k < N_FREQ; k++) {
        double pA = dft_A_re[k]*dft_A_re[k]+dft_A_im[k]*dft_A_im[k];
        double pB = dft_B_re[k]*dft_B_re[k]+dft_B_im[k]*dft_B_im[k];
        lp_E += fabs(pA+pB-TARGET);
    }
    return lp_E;
}

int main(int argc, char **argv) {
    int n_trials = argc>1 ? atoi(argv[1]) : 100;
    long long max_iter = argc>2 ? atoll(argv[2]) : 50000000LL;
    uint64_t base_seed = argc>3 ? atoll(argv[3]) : 42;

    fprintf(stderr, "Phase-space search: %d trials x %lldM iters\n",
            n_trials, max_iter/1000000);

    double global_best_lp = 1e18;
    int global_A[LEN], global_B[LEN];

    for (int trial = 0; trial < n_trials; trial++) {
        seed_rng(base_seed + trial*7919);

        /* Initialize random t, phi, psi */
        double t[N_FREQ], phi[N_FREQ], psi[N_FREQ];
        for (int k = 0; k < N_FREQ; k++) {
            t[k] = rnd() * TARGET;
            phi[k] = rnd() * 2*PI;
            psi[k] = rnd() * 2*PI;
        }

        double E = energy(t, phi, psi);
        double best_E = E;
        double best_t[N_FREQ], best_phi[N_FREQ], best_psi[N_FREQ];
        memcpy(best_t, t, sizeof(t));
        memcpy(best_phi, phi, sizeof(phi));
        memcpy(best_psi, psi, sizeof(psi));

        /* SA in phase space */
        double T_init = E * 0.05;
        for (long long it = 0; it < max_iter; it++) {
            double progress = (double)it / max_iter;
            double T = T_init * pow(1.0-progress, 1.5);
            if (T < 1e-6) T = 1e-6;

            /* Copy state */
            double nt[N_FREQ], np[N_FREQ], ns[N_FREQ];
            memcpy(nt, t, sizeof(t));
            memcpy(np, phi, sizeof(phi));
            memcpy(ns, psi, sizeof(psi));

            /* Random perturbation */
            int idx = (int)(rnd() * N_FREQ);
            double r = rnd();
            if (r < 0.33) {
                /* Perturb t */
                nt[idx] += (rnd()-0.5) * 40.0 * (1.0-progress);
                if (nt[idx] < 0) nt[idx] = 0;
                if (nt[idx] > TARGET) nt[idx] = TARGET;
            } else if (r < 0.67) {
                /* Perturb phi */
                np[idx] += (rnd()-0.5) * 0.5 * (1.0-progress);
            } else {
                /* Perturb psi */
                ns[idx] += (rnd()-0.5) * 0.5 * (1.0-progress);
            }

            double nE = energy(nt, np, ns);
            double dE = nE - E;
            if (dE < 0 || rnd() < exp(-dE/T)) {
                memcpy(t, nt, sizeof(t));
                memcpy(phi, np, sizeof(phi));
                memcpy(psi, ns, sizeof(psi));
                E = nE;
                if (E < best_E) {
                    best_E = E;
                    memcpy(best_t, t, sizeof(t));
                    memcpy(best_phi, phi, sizeof(phi));
                    memcpy(best_psi, psi, sizeof(psi));
                }
            }
        }

        /* Round and check LP */
        int rA[LEN], rB[LEN];
        double lp_E = round_and_check(best_t, best_phi, best_psi, rA, rB);

        fprintf(stderr, "Trial %d: phase_E=%.3f -> LP_E=%.1f\n",
                trial, best_E, lp_E);

        if (lp_E < global_best_lp) {
            global_best_lp = lp_E;
            memcpy(global_A, rA, sizeof(int)*LEN);
            memcpy(global_B, rB, sizeof(int)*LEN);
        }

        if (lp_E < 0.5) {
            fprintf(stderr, "\n*** EXACT SOLUTION ***\n");
            printf("A = [");
            for(int i=0;i<LEN;i++) printf("%d%s",rA[i],i<LEN-1?",":"");
            printf("]\nB = [");
            for(int i=0;i<LEN;i++) printf("%d%s",rB[i],i<LEN-1?",":"");
            printf("]\nENERGY = %.9f\n", lp_E);
            return 0;
        }
    }

    fprintf(stderr, "\nBest LP energy: %.1f\n", global_best_lp);
    printf("BEST_A = [");
    for(int i=0;i<LEN;i++) printf("%d%s",global_A[i],i<LEN-1?",":"");
    printf("]\nBEST_B = [");
    for(int i=0;i<LEN;i++) printf("%d%s",global_B[i],i<LEN-1?",":"");
    printf("]\nENERGY = %.9f\n", global_best_lp);
    return 0;
}
