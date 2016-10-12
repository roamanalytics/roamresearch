#include <Rcpp.h>
using namespace Rcpp;

//' Grubbs' Test
//'
//' @param y NumericVector
//' @param n IntegerVector
//' @param alpha double
//' @export
// [[Rcpp::export]]
NumericVector grubbs_test_cpp(NumericVector y,
                              IntegerVector n,
                              double alpha) {
  // Prep
  int N = y.length();

  // Rcout << "N " << N << std::endl;

  double y_bar = Rcpp::mean(y);
  double s = Rcpp::sd(y);
  NumericVector diff = Rcpp::abs(y - y_bar);
  double numer = Rcpp::max(diff);

  // Current element
  int cur_max_ind = Rcpp::which_max(diff);
  int out_max_ind = n[cur_max_ind];
  double max_val = y[cur_max_ind];

  // Grubbs
  double G = numer / s;

  // Rcout << "G " << G << std::endl;

  NumericVector alpha_v = NumericVector::create(alpha / (2*N));
  double t = Rcpp::qt(alpha_v, N-2)[0];
  double t_2 = t * t;
  double critical_value = ((N - 1) / sqrt(N)) * sqrt(t_2/(N - 2 + t_2));

  // Rcout << "critical_value " << critical_value << std::endl;

  int result;
  if (G > critical_value) {
    result = 1;
  } else {
    result = 0;
  }

  NumericVector output = NumericVector::create(out_max_ind,
                                               cur_max_ind,
                                               result);
  return output;
}

//' Sequential Grubbs' Tests
//'
//' @param in_vector NumericVector
//' @param max_outlier_pct double
//' @param alpha double
//' @export
// [[Rcpp::export]]
IntegerVector sequential_grubbs_test_cpp(NumericVector in_vector,
                                         double max_outlier_pct,
                                         double alpha) {
  // Input prep
  int nrow = in_vector.length();
  IntegerVector in_vector_ind = seq_len(nrow) - 1; // Need 0 index
  int max_outlier_count = (int) nrow * max_outlier_pct;

  // Sequential Grubbs tests
  NumericVector out_max_inds = NumericVector(max_outlier_count);
  NumericVector results = NumericVector(max_outlier_count);

  for (int i = 0; i < max_outlier_count; i++) {
    // Test
    NumericVector grubbs_result = grubbs_test_cpp(in_vector,
                                                  in_vector_ind,
                                                  alpha);
    int out_max_ind = grubbs_result[0];
    int cur_max_ind = grubbs_result[1];
    int result = grubbs_result[2];

    // Save
    out_max_inds[i] = out_max_ind;
    results[i] = result;

    // Pop
    in_vector.erase(cur_max_ind);
    in_vector_ind.erase(cur_max_ind);
  }

  // Replace
  for (int i = 0; i < max_outlier_count; i++) {
    if (results[i] == 1) {
      for (int j = 0; j < i+1; j++) {
        results[j] = 1;
      }
    }
  }

  // Output
  IntegerVector return_vector(nrow);
  for (int i = 0; i < max_outlier_count; i++) {
    return_vector(out_max_inds[i]) = results[i];
  }

  return return_vector;
}

//' Sequential Grubbs' Tests Df
//'
//' @param df DataFrame
//' @param var_in string
//' @param var_out string
//' @param max_outlier_pct double
//' @param alpha double
//' @export
// [[Rcpp::export]]
DataFrame sequential_grubbs_test_cpp_df(DataFrame df,
                                        std::string var_in,
                                        std::string var_out,
                                        double max_outlier_pct,
                                        double alpha) {
  // Input prep
  NumericVector in_vector = df[var_in];
  IntegerVector result = sequential_grubbs_test_cpp(in_vector,
                                                    max_outlier_pct,
                                                    alpha);
  df[var_out] = result;
  return df;
}

// [[Rcpp::interfaces(r, cpp)]]
