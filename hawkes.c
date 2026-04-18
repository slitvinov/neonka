/* hawkes.c — 8-D mutual-excitation Hawkes MLE via EM (Ogata).
 *
 * Model:
 *   λ_c(t) = μ_c + Σ_j α_{c,j} · φ_j(t)
 *   φ_j(t) = Σ_{t_k^j < t} exp(-β (t - t_k^j))
 *
 * Input  (stdin): packed events {int32 t, int32 type} in time order.
 * Output (stdout): one record per line:
 *     beta <value>
 *     mu  <c> <value>
 *     alpha <c> <j> <value>
 *
 * Usage: hawkes [-i max_iter] [-b beta] [-t tol]
 *   -i N     max EM iterations (default 80)
 *   -b B     fixed β (default 0.05)
 *   -t TOL   stop if |Δ log L| < TOL (default 1e-4)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 10 };

struct Event { int32_t t, type; };

static double eval_ll(double b, const struct Event *ev, size_t n,
                      int32_t t0, int32_t t1, double T_total,
                      const double *mu, const double alpha[D][D]) {
  double phi[D] = {0};
  double lls = 0, G[D] = {0};
  int32_t last = t0;
  for (size_t i = 0; i < n; i++) {
    int32_t dt = ev[i].t - last;
    if (dt > 0) { double dec = exp(-b * dt);
      for (int j = 0; j < D; j++) phi[j] *= dec; }
    int c = ev[i].type;
    double lam = mu[c];
    for (int j = 0; j < D; j++) lam += alpha[c][j] * phi[j];
    if (lam < 1e-12) lam = 1e-12;
    lls += log(lam);
    G[c] += (1.0 - exp(-b * (double)(t1 - ev[i].t))) / b;
    phi[c] += 1.0;
    last = ev[i].t;
  }
  double comp = 0;
  for (int c = 0; c < D; c++) {
    comp += mu[c] * T_total;
    for (int j = 0; j < D; j++) comp += alpha[c][j] * G[j];
  }
  (void)n;  /* suppress unused param when compiled as -O0 */
  return lls - comp;
}

int main(int argc, char **argv) {
  int max_iter = 80;
  double beta = 0.05, tol = 1e-4;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i+1 < argc) max_iter = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-b") && i+1 < argc) beta = atof(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i+1 < argc) tol = atof(argv[++i]);
    else { fprintf(stderr, "hawkes: unknown arg '%s'\n", argv[i]); return 1; }
  }

  size_t cap = 1 << 20, n = 0;
  struct Event *ev = malloc(cap * sizeof *ev);
  while (1) {
    if (n == cap) { cap *= 2; ev = realloc(ev, cap * sizeof *ev); }
    if (fread(&ev[n], sizeof *ev, 1, stdin) != 1) break;
    n++;
  }
  if (n == 0) { fprintf(stderr, "hawkes: no events\n"); return 1; }
  fprintf(stderr, "hawkes: loaded %zu events\n", n);

  int32_t t0 = ev[0].t, t1 = ev[n-1].t + 1;
  double T_total = (double)(t1 - t0);

  int N_j[D] = {0};
  for (size_t i = 0; i < n; i++) N_j[ev[i].type]++;

  double mu[D], alpha[D][D];
  for (int c = 0; c < D; c++) {
    mu[c]    = 0.5 * (double)N_j[c] / T_total;
    for (int j = 0; j < D; j++) alpha[c][j] = 0.01;
  }

  double ll_prev = -INFINITY;
  for (int iter = 0; iter < max_iter; iter++) {
    double mu_num[D] = {0};
    double alpha_num[D][D] = {{0}};
    double phi[D] = {0};
    double log_lambda_sum = 0;
    double G[D] = {0};
    int32_t last_t = t0;

    for (size_t i = 0; i < n; i++) {
      int32_t dt = ev[i].t - last_t;
      if (dt > 0) {
        double dec = exp(-beta * dt);
        for (int j = 0; j < D; j++) phi[j] *= dec;
      }
      int c = ev[i].type;
      double lambda = mu[c];
      for (int j = 0; j < D; j++) lambda += alpha[c][j] * phi[j];
      if (lambda < 1e-12) lambda = 1e-12;
      log_lambda_sum += log(lambda);

      mu_num[c] += mu[c] / lambda;
      for (int j = 0; j < D; j++)
        alpha_num[c][j] += alpha[c][j] * phi[j] / lambda;

      G[c] += (1.0 - exp(-beta * (double)(t1 - ev[i].t))) / beta;

      phi[c] += 1.0;
      last_t = ev[i].t;
    }

    double compensator = 0;
    for (int c = 0; c < D; c++) {
      compensator += mu[c] * T_total;
      for (int j = 0; j < D; j++) compensator += alpha[c][j] * G[j];
    }
    double ll = log_lambda_sum - compensator;

    for (int c = 0; c < D; c++) {
      mu[c] = mu_num[c] / T_total;
      for (int j = 0; j < D; j++)
        if (G[j] > 0) alpha[c][j] = alpha_num[c][j] / G[j];
    }

    fprintf(stderr, "iter %3d  log L = %.4f  Δ = %+.4e  β=%.4f\n",
            iter, ll, ll - ll_prev, beta);
    if (iter > 0 && fabs(ll - ll_prev) < tol * fabs(ll)) {
      fprintf(stderr, "converged on (μ,α); now optimizing β\n");
      break;
    }
    ll_prev = ll;
  }

  /* Golden-section search on β ∈ [1e-3, 1] */
  {
    double lo = 0.001, hi = 1.0;
    double gr = (sqrt(5.0) - 1.0) / 2.0;
    double b_c = hi - gr * (hi - lo), b_d = lo + gr * (hi - lo);
    double ll_c = eval_ll(b_c, ev, n, t0, t1, T_total, mu, alpha);
    double ll_d = eval_ll(b_d, ev, n, t0, t1, T_total, mu, alpha);
    for (int k = 0; k < 30; k++) {
      if (ll_c > ll_d) { hi = b_d; b_d = b_c; ll_d = ll_c;
        b_c = hi - gr * (hi - lo);
        ll_c = eval_ll(b_c, ev, n, t0, t1, T_total, mu, alpha); }
      else { lo = b_c; b_c = b_d; ll_c = ll_d;
        b_d = lo + gr * (hi - lo);
        ll_d = eval_ll(b_d, ev, n, t0, t1, T_total, mu, alpha); }
      if (hi - lo < 1e-6) break;
    }
    beta = 0.5 * (lo + hi);
    fprintf(stderr, "β-opt converged: β=%.6f  log L=%.4f\n",
            beta, eval_ll(beta, ev, n, t0, t1, T_total, mu, alpha));
  }

  printf("beta %.6f\n", beta);
  for (int c = 0; c < D; c++) printf("mu %d %.8f\n", c, mu[c]);
  for (int c = 0; c < D; c++)
    for (int j = 0; j < D; j++)
      printf("alpha %d %d %.8f\n", c, j, alpha[c][j]);

  free(ev);
  return 0;
}
