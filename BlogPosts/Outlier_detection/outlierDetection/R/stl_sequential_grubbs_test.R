#' STL Sequential Grubbs' Tests
#'
#' @param series numeric vector for time series, can't have NA values
#' @param frequency_season string of "minute_day", "hour_day", "hour_week",
#' "day_week", "day_year", "week_year", "month_year"
#' @param max_outlier_pct double
#' @param alpha double
#' @export
stl_sequential_grubbs_test <- function(series,
                                       frequency_season,
                                       max_outlier_pct=0.05,
                                       alpha=0.05) {
  if (frequency_season == 'minute_day') {
    freq <- 60*24
  } else if (frequency_season == 'hour_day') {
    freq <- 24
  } else if (frequency_season == 'hour_week') {
    freq <- 24*7
  } else if (frequency_season == 'day_week') {
    freq <- 7
  } else if (frequency_season == 'day_year') {
    freq <- 365
  } else if (frequency_season == 'week_year') {
    freq <- 52
  } else if (frequency_season == 'month_year') {
    freq <- 12
  }

  ts <- ts(series, frequency=freq)
  stl <- stl(ts, s.window='periodic')
  remainder <- as.numeric(stl$time.series[,'remainder'])
  outliers <- sequential_grubbs_test_cpp(remainder,
                                         max_outlier_pct=max_outlier_pct,
                                         alpha=alpha)

  return(outliers)
}


#' STL Sequential Grubbs' Tests DF
#'
#' @param df DataFrame
#' @param var_in string name of input column, can't have NA values
#' @param var_out string name of output column
#' @param frequency_season String of "minute_day", "hour_day", "hour_week",
#' "day_week", "day_year", "week_year", "month_year"
#' @param max_outlier_pct double
#' @param alpha double
#' @export
stl_sequential_grubbs_test_df <- function(df,
                                          var_in,
                                          var_out,
                                          frequency_season,
                                          max_outlier_pct=0.05,
                                          alpha=0.05) {
  if (frequency_season == 'minute_day') {
    freq <- 60*24
  } else if (frequency_season == 'hour_day') {
    freq <- 24
  } else if (frequency_season == 'hour_week') {
    freq <- 24*7
  } else if (frequency_season == 'day_week') {
    freq <- 7
  } else if (frequency_season == 'day_year') {
    freq <- 365
  } else if (frequency_season == 'week_year') {
    freq <- 52
  } else if (frequency_season == 'month_year') {
    freq <- 12
  }

  series <- df[[var_in]]
  ts <- ts(series, frequency=freq)
  stl <- stl(ts, s.window='periodic')
  remainder <- as.numeric(stl$time.series[,'remainder'])
  outliers <- sequential_grubbs_test_cpp(remainder,
                                         max_outlier_pct=max_outlier_pct,
                                         alpha=alpha)
  outliers_df <- data.frame(tmp = outliers)
  names(outliers_df) <- var_out

  out_df <- cbind(df, outliers_df)
  return(out_df)
}
