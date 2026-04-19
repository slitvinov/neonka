/* hawkes.c — 6-D Hawkes MLE with single-exponential kernel (β=0.05 fixed).
 *
 * Model:
 *   λ_c(t) = μ_c + Σ_j α_{c,j} · φ_j(t)
 *   φ_j(t) = Σ_{t_i^j < t} exp(-β · (t - t_i^j))
 *
 * β = 0.05 (half-life ≈ 14 ticks) matches the event-clustering timescale
 * that actually modulates rates over our prediction horizons T≤55.  Earlier
 * 3-exponential mixture (β∈{0.5, 0.05, 0.005}) added two kernels that
 * either decay faster than T=1 or slower than T=55 — neither informative —
 * so 3× the parameters for no observable benefit in downstream metrics.
 *
 * Stability: after each EM M-step, rescale each α row so that
 *   ρ_c := Σ_j α[c,j]/β  ≤  RHO_MAX  (default 0.90)
 * By Gershgorin, max_c ρ_c ≥ spectral radius of branching matrix.
 *
 * Input  (stdin): packed (int32 t, int32 type) in time order; types 0..5.
 * Output (stdout):
 *     beta 0 <value>
 *     mu <c> <value>
 *     alpha <c> <j> <value>
 *
 * Usage: hawkes [-i N] [-t TOL] [-r RHO_MAX]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 6 };
static const double BETA = 0.05;

struct Event { int32_t t, type; };

int main(int argc, char **argv) {
  int max_iter = 300;
  double tol = 1e-6, rho_max = 0.90;
  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-i") && i + 1 < argc) max_iter = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) tol      = atof(argv[++i]);
    else if (!strcmp(argv[i], "-r") && i + 1 < argc) rho_max  = atof(argv[++i]);
    else { fprintf(stderr, "hawkes: unknown arg '%s'\n", argv[i]); return 1; }
  }
  fprintf(stderr, "hawkes: D=%d β=%g ρ_max=%.2f\n", D, BETA, rho_max);

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

  /* Init: α/β = 0.005 uniform → ρ_c ≈ 0.03·6 = 0.18. */
  double mu[D];
  double alpha[D][D];
  for (int c = 0; c < D; c++) {
    mu[c] = 0.5 * (double)N_j[c] / T_total;
    for (int j = 0; j < D; j++) alpha[c][j] = 0.005 * BETA;
  }

  double phi[D], mu_num[D];
  double alpha_num[D][D], G[D];

  double ll_prev = -INFINITY;
  for (int iter = 0; iter < max_iter; iter++) {
    memset(phi, 0, sizeof phi);
    memset(mu_num, 0, sizeof mu_num);
    memset(alpha_num, 0, sizeof alpha_num);
    memset(G, 0, sizeof G);
    double log_lambda_sum = 0;
    int32_t last_t = t0;

    for (size_t i = 0; i < n; i++) {
      int32_t dt = ev[i].t - last_t;
      if (dt > 0) {
        double dec = exp(-BETA * dt);
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

      G[c] += (1.0 - exp(-BETA * (double)(t1 - ev[i].t))) / BETA;
      phi[c] += 1.0;
      last_t = ev[i].t;
    }

    double comp = 0;
    for (int c = 0; c < D; c++) {
      comp += mu[c] * T_total;
      for (int j = 0; j < D; j++) comp += alpha[c][j] * G[j];
    }
    double ll = log_lambda_sum - comp;

    /* M-step */
    for (int c = 0; c < D; c++) {
      mu[c] = mu_num[c] / T_total;
      for (int j = 0; j < D; j++)
        if (G[j] > 0) alpha[c][j] = alpha_num[c][j] / G[j];
    }

    /* Stability: row-wise branching clamp. Ensures ρ ≤ rho_max via Gershgorin. */
    double max_row_rho = 0;
    for (int c = 0; c < D; c++) {
      double row_rho = 0;
      for (int j = 0; j < D; j++) row_rho += alpha[c][j] / BETA;
      if (row_rho > max_row_rho) max_row_rho = row_rho;
      if (row_rho > rho_max) {
        double scale = rho_max / row_rho;
        for (int j = 0; j < D; j++) alpha[c][j] *= scale;
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
  printf("beta 0 %g\n", BETA);
  for (int c = 0; c < D; c++) printf("mu %d %g\n", c, mu[c]);
  for (int c = 0; c < D; c++)
    for (int j = 0; j < D; j++)
      printf("alpha %d %d %g\n", c, j, alpha[c][j]);

  free(ev);
  return 0;
}
