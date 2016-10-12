library(outlierDetection)
context("STL Sequential Grubbs' test")

test_that("stl_sequential_grubbs_test is accurate on a series", {
  x <- -100:100
  y1 <- 100*sin(x/(24 / (2*pi)))
  y2 <- 0.05*x^2
  y <- y1+y2
  y[10] <- 700
  y[50] <- 200
  y[100] <- -200
  y[150] <- 400
  y[190] <- 0
  outliers <- stl_sequential_grubbs_test(y, frequency_season='hour_day'
                                         , max_outlier_pct=0.10, alpha=0.05)
  expect_equal(outliers[10], 1)
  expect_equal(outliers[50], 1)
  expect_equal(outliers[100], 1)
  expect_equal(outliers[150], 1)
  expect_equal(outliers[190], 1)
})
