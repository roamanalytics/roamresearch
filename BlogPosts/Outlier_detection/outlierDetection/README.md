# outlierDetection

## Overview

This package performs a method of outlier detection by

1. using STL decomposition to define a time series pattern, and
2. applying sequential Grubbs' tests to spot points that don't fit that pattern.

## Installation

- devtools
- outlierDetection (use install_github from devtools)

## Usage

### R

Assuming series is monthly data with a yearly season:
```R
library(outlierDetection)

outlier <- stl_sequential_grubbs_test(
    series, 'month_year', max_outlier_pct=0.05, alpha=0.05)
```

Or if you have series in a dataframe (with dplyr):
```R
library(outlierDetection)
library(dplyr)

outlier_df <- df %>%
    mutate(outlier = stl_sequential_grubbs_test(
            series, 'month_year', max_outlier_pct=0.05, alpha=0.05))
```

Or if you have many series stacked in a dataframe made distinct by group_var
(with multidplyr and parallel):
```R
library(outlierDetection)
library(multidplyr)
library(parallel)

clus <- create_cluster(detectCores())
clus %>% cluster_library("outlierDetection")
outlier_df <- df %>%
    partition(group_var, cluster=clus) %>%
    mutate(outlier = stl_sequential_grubbs_test(
            series, 'month_year', max_outlier_pct=0.05, alpha=0.05)) %>%
    collect() %>%
    ungroup()
```

### Python (to R)

Easiest to pass back and forth using a DataFrame (pandas and rpy2):
```Python
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
outlier = importr('outlierDetection')

pandas2ri.activate()
tmp_df = outlier.stl_sequential_grubbs_test_df(
    df=df, var_in='series', var_out='outlier',
    frequency_season='month_year',
    max_outlier_pct=0.05, alpha=0.05)
outlier_df = pandas2ri.ri2py(tmp_df)
pandas2ri.deactivate()
```
