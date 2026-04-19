/* hawkes.c — 6-D Hawkes MLE with K-exponential mixture kernel.
 *
 * Model:
 *   λ_c(t) = μ_c + Σ_j Σ_k α_{c,j,k} · φ_{j,k}(t)
 *   φ_{j,k}(t) = Σ_{t_i^j < t} exp(-β_k · (t - t_i^j))
 *
 * β_k geometrically spaced {0.5, 0.05, 0.005} — approximates power-law kernel
 * (Bacry-Jaisson 2016) over 3 decades of lag.
 *
 * Stability: after each EM M-step, rescale each α row so that
 *   ρ_c := Σ_j Σ_k α[c,j,k]/β_k  ≤  RHO_MAX  (default 0.90)
 * By Gershgorin, max_c ρ_c ≥ spectral radius of effective branching matrix,
 * so row-wise clamping bounds ρ(B) ≤ RHO_MAX strictly.
 *
 * Input  (stdin): packed (int32 t, int32 type) in time order; types 0..5.
 * Output (stdout):
 *     beta <k> <value>
 *     mu <c> <value>
 *     alpha <c> <j> <k> <value>
 *
 * Usage: hawkes [-i N] [-t TOL] [-r RHO_MAX]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 6, K = 3 };
static const double BETAS[K] = { 0.5, 0.05, 0.005 };

struct Event { int32_t t, type; };

int main(int argc, char **argv) {
  int max_iter = 300;
  double tol = 1e-6, rho_max = 0.90;
  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-i") && i + 1 < argc) max_iter = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) tol      = atof(argv[++i]);
    else if (!strcmp(argv[i], "-r") && i + 1 < argc) rho_max  = atof(argv[++i]);
    else if (!strcmp(argv[i], "-b") && i + 1 < argc) (void)atof(argv[++i]); /* legacy */
    else { fprintf(stderr, "hawkes: unknown arg '%s'\n", argv[i]); return 1; }
  }
  fprintf(stderr, "hawkes: D=%d K=%d β={%.3f,%.3f,%.3f} ρ_max=%.2f\n",
          D, K, BETAS[0], BETAS[1], BETAS[2], rho_max);

  size_t cap = 1 << 20, n = 0;
  struct Event *ev = malloc(cap * sizeof *ev);
  struct Event tmp;
  while (fread(&tmp, sizeof tmp, 1, stdin) == 1) {
    if (tmp.type < 0 || tmp.type >= D) continue;
    if (n == cap) { cap *= 2; ev = realloc(ev, cap * sizeof *ev); }
    ev[n++] = tmp;
  }
  if (n == 0) { fprintf(stderr, "hawkes: no events\n"); return 1; }

  int32_t t0 = ev[0].t, t1 = ev[n-1].t + 1;
  double T_total = (double)(t1 - t0);

  int N_j[D] = {0};
  for (size_t i = 0; i < n; i++) N_j[ev[i].type]++;
  fprintf(stderr, "hawkes: %zu events T=%g rate=%.4f\n",
          n, T_total, (double)n / T_total);

  /* Init: small α, spread roughly equally across k; gives initial ρ ≈ 0.3. */
  double mu[D];
  static double alpha[D][D][K];
  for (int c = 0; c < D; c++) {
    mu[c] = 0.5 * (double)N_j[c] / T_total;
    for (int j = 0; j < D; j++)
      for (int k = 0; k < K; k++)
        alpha[c][j][k] = 0.005 * BETAS[k];   /* α/β = 0.005 uniform */
  }

  static double phi[D][K], mu_num[D];
  static double alpha_num[D][D][K], G[D][K];

  double ll_prev = -INFINITY;
  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    memset(phi, 0, sizeof phi);
    memset(mu_num, 0, sizeof mu_num);
    memset(alpha_num, 0, sizeof alpha_num);
    memset(G, 0, sizeof G);
    double log_lambda_sum = 0;
    int32_t last_t = t0;

    for (size_t i = 0; i < n; i++) {
      int32_t dt = ev[i].t - last_t;
      if (dt > 0)
        for (int k = 0; k < K; k++) {
          double dec = exp(-BETAS[k] * dt);
          for (int j = 0; j < D; j++) phi[j][k] *= dec;
        }
      int c = ev[i].type;
      double lambda = mu[c];
      for (int j = 0; j < D; j++)
        for (int k = 0; k < K; k++) lambda += alpha[c][j][k] * phi[j][k];
      if (lambda < 1e-12) lambda = 1e-12;
      log_lambda_sum += log(lambda);

      mu_num[c] += mu[c] / lambda;
      for (int j = 0; j < D; j++)
        for (int k = 0; k < K; k++)
          alpha_num[c][j][k] += alpha[c][j][k] * phi[j][k] / lambda;

      for (int k = 0; k < K; k++)
        G[c][k] += (1.0 - exp(-BETAS[k] * (double)(t1 - ev[i].t))) / BETAS[k];
      for (int k = 0; k < K; k++) phi[c][k] += 1.0;
      last_t = ev[i].t;
    }

    double comp = 0;
    for (int c = 0; c < D; c++) {
      comp += mu[c] * T_total;
      for (int j = 0; j < D; j++)
        for (int k = 0; k < K; k++) comp += alpha[c][j][k] * G[j][k];
    }
    double ll = log_lambda_sum - comp;

    /* M-step */
    for (int c = 0; c < D; c++) {
      mu[c] = mu_num[c] / T_total;
      for (int j = 0; j < D; j++)
        for (int k = 0; k < K; k++)
          if (G[j][k] > 0) alpha[c][j][k] = alpha_num[c][j][k] / G[j][k];
    }

    /* Stability: row-wise branching clamp. Ensures ρ ≤ rho_max via Gershgorin. */
    double max_row_rho = 0;
    for (int c = 0; c < D; c++) {
      double row_rho = 0;
      for (int j = 0; j < D; j++)
        for (int k = 0; k < K; k++)
          row_rho += alpha[c][j][k] / BETAS[k];
      if (row_rho > max_row_rho) max_row_rho = row_rho;
      if (row_rho > rho_max) {
        double scale = rho_max / row_rho;
        for (int j = 0; j < D; j++)
          for (int k = 0; k < K; k++) alpha[c][j][k] *= scale;
      }
    }

    fprintf(stderr, "iter %3d  ll=%.4f  Δ=%+.4e  max_row_ρ=%.4f\n",
            iter, ll, ll - ll_prev, max_row_rho);
    if (iter > 0 && fabs(ll - ll_prev) < tol * fabs(ll)) {
      fprintf(stderr, "converged\n");
      break;
    }
    ll_prev = ll;
  }

  /* Output */
  for (int k = 0; k < K; k++) printf("beta %d %g\n", k, BETAS[k]);
  for (int c = 0; c < D; c++) printf("mu %d %g\n", c, mu[c]);
  for (int c = 0; c < D; c++)
    for (int j = 0; j < D; j++)
      for (int k = 0; k < K; k++)
        printf("alpha %d %d %d %g\n", c, j, k, alpha[c][j][k]);

  free(ev);
  return 0;
}
