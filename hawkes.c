/* hawkes.c — 6-D pooled Hawkes MLE, single-exponential kernel (β=0.05).
 *
 * Model:
 *   λ_c(t) = μ_c + Σ_j α_{c,j} · φ_j(t)
 *   φ_j(t) = Σ_{t_i^j < t} exp(-β (t - t_i^j))
 *
 * Input  (stdin): packed events {int32 t, int32 type} in time order. Types
 *   pre-pooled (ask+bid merged) and tm-split by preproc_events.py.
 * Output (stdout):
 *     beta <value>
 *     mu <c> <value>
 *     alpha <c> <j> <value>
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 6 };

struct Event { int32_t t, type; };

int main(int argc, char **argv) {
  int max_iter = 500;
  double beta = 0.05, tol = 1e-7;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i+1 < argc) max_iter = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-b") && i+1 < argc) beta = atof(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i+1 < argc) tol = atof(argv[++i]);
    else { fprintf(stderr, "hawkes: unknown arg '%s'\n", argv[i]); return 1; }
  }

  size_t cap = 1 << 20, n = 0;
  struct Event *ev = malloc(cap * sizeof *ev);
  struct Event tmp;
  while (1) {
    if (fread(&tmp, sizeof tmp, 1, stdin) != 1) break;
    if (tmp.type < 0 || tmp.type >= D) continue;
    if (n == cap) { cap *= 2; ev = realloc(ev, cap * sizeof *ev); }
    ev[n++] = tmp;
  }
  if (n == 0) { fprintf(stderr, "hawkes: no events\n"); return 1; }
  fprintf(stderr, "hawkes: %zu events, D=%d (pooled, tm split)\n", n, D);

  int32_t t0 = ev[0].t, t1 = ev[n-1].t + 1;
  double T_total = (double)(t1 - t0);

  int N_j[D] = {0};
  for (size_t i = 0; i < n; i++) N_j[ev[i].type]++;

  double mu[D], alpha[D][D];
  for (int c = 0; c < D; c++) {
    mu[c] = 0.5 * (double)N_j[c] / T_total;
    for (int j = 0; j < D; j++) alpha[c][j] = 0.01;
  }

  double ll_prev = -INFINITY;
  for (int iter = 0; iter < max_iter; iter++) {
    double mu_num[D] = {0}, alpha_num[D][D] = {{0}};
    double phi[D] = {0}, G[D] = {0};
    double log_lambda_sum = 0;
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
      for (int j = 0; j < D; j++) alpha_num[c][j] += alpha[c][j] * phi[j] / lambda;
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
    fprintf(stderr, "iter %3d  ll=%.4f  Δ=%+.4e\n", iter, ll, ll - ll_prev);
    if (iter > 0 && fabs(ll - ll_prev) < tol * fabs(ll)) {
      fprintf(stderr, "converged\n");
      break;
    }
    ll_prev = ll;
  }

  printf("beta %.6f\n", beta);
  for (int c = 0; c < D; c++) printf("mu %d %.8f\n", c, mu[c]);
  for (int c = 0; c < D; c++)
    for (int j = 0; j < D; j++)
      printf("alpha %d %d %.8f\n", c, j, alpha[c][j]);

  free(ev);
  return 0;
}
