// Hierarchical Bayesian trust-update model for the investor in MRTT.
// Belief about the fraction theta of the tripled amount the trustee returns: Beta(a, b).
// After each round with inv > 0 the belief is updated with the observed fraction k/4 (weight eta).
// Choice: softmax over investments 0..20 with utility z*(3m-1) - rho*9*v*z^2, z = x/20,
// where m and v are the belief mean and variance (mean-variance trade-off).
data {
  int<lower=1> N;
  int<lower=1> T;
  array[N, T] int<lower=0, upper=20> inv;
  array[N, T] int<lower=0, upper=4> k;
}
transformed data {
  real kappa0 = 2;
  vector[21] z;
  for (x in 0:20) z[x + 1] = x / 20.0;
}
parameters {
  vector[4] mu_g;
  vector<lower=0>[4] sd_g;
  matrix[N, 4] raw;
}
transformed parameters {
  vector[N] mu0 = inv_logit(mu_g[1] + sd_g[1] * raw[:, 1]);
  vector[N] eta = exp(mu_g[2] + sd_g[2] * raw[:, 2]);
  vector[N] beta = exp(mu_g[3] + sd_g[3] * raw[:, 3]);
  vector[N] rho = mu_g[4] + sd_g[4] * raw[:, 4];
}
model {
  mu_g[1] ~ normal(0, 1);
  mu_g[2] ~ normal(0, 0.7);
  mu_g[3] ~ normal(log(3), 0.7);
  mu_g[4] ~ normal(0, 1);
  sd_g ~ normal(0, 0.5);
  to_vector(raw) ~ std_normal();

  for (i in 1:N) {
    real a = kappa0 * mu0[i];
    real b = kappa0 * (1 - mu0[i]);
    for (t in 1:T) {
      real m = a / (a + b);
      real v = a * b / ((a + b) ^ 2 * (a + b + 1));
      vector[21] logits = beta[i] * (z * (3 * m - 1) - rho[i] * 9 * v * square(z));
      inv[i, t] + 1 ~ categorical_logit(logits);
      if (inv[i, t] > 0) {
        real r = k[i, t] / 4.0;
        a += eta[i] * r;
        b += eta[i] * (1 - r);
      }
    }
  }
}
generated quantities {
  array[N, T] real log_lik;
  array[N, T] int inv_rep;
  for (i in 1:N) {
    real a = kappa0 * mu0[i];
    real b = kappa0 * (1 - mu0[i]);
    for (t in 1:T) {
      real m = a / (a + b);
      real v = a * b / ((a + b) ^ 2 * (a + b + 1));
      vector[21] logits = beta[i] * (z * (3 * m - 1) - rho[i] * 9 * v * square(z));
      log_lik[i, t] = categorical_logit_lpmf(inv[i, t] + 1 | logits);
      inv_rep[i, t] = categorical_logit_rng(logits) - 1;
      if (inv[i, t] > 0) {
        real r = k[i, t] / 4.0;
        a += eta[i] * r;
        b += eta[i] * (1 - r);
      }
    }
  }
}
