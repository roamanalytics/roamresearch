library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Setup
roam_palette <- c('#0499CC', '#4D8951', '#FDBA58', '#A3AF98'
                  , '#8F8F43', '#838A8A', '#876DB5', '#735E56'
                  , '#607D8B', '#32A8B4', '#9BB8D7', '#212121'
                  , '#B8B8B8', '#03A9F4')

# Example STL Plot
x <- -100:100
y <- 100*sin(x/(24 / (2*pi))) + 0.05*x^2
ts <- ts(y, freq=24)
r <- stl(ts, s.window='periodic')
res <- data_frame(index=seq_along(x)
                  , data=as.numeric(y)
                  , seasonal=as.numeric(r$time.series[,1])
                  , trend=as.numeric(r$time.series[,2])
                  , remainder=as.numeric(r$time.series[,3])) %>%
  gather(component, value, -index) %>%
  mutate(component=factor(component, levels=c("data","trend","seasonal","remainder")))

stl_pic <- ggplot(res, aes(x=index, y=value, color=component)) +
  geom_line() +
  facet_grid(component ~ ., scales='fixed') +
  ggtitle("STL Decomposition") +
  ylab("") +
  scale_color_manual(values=roam_palette) +
  theme_minimal() +
  theme(legend.position="none"
        , panel.border = element_rect(color = "grey", fill=NA, size=0.2))
ggsave("./fig/stl_pic.png", plot=stl_pic)

# All Series Plot
df <- read_csv("./data/long_df_ABC.csv")

all_series <- ggplot(df, aes(x=dt, y=value, color=series)) +
  geom_line() +
  facet_grid(series ~ ., scales='free_y') +
  ggtitle("All series") +
  xlab("") + ylab("") +
  scale_color_manual(values=roam_palette) +
  theme_minimal() +
  theme(legend.position="none"
        , panel.border = element_rect(color = "grey", fill=NA, size=0.2))
ggsave("./fig/all_series.png", plot=all_series)

# Outlier Plots
df <- read_csv("./data/wide_outlier_df_ABC.csv") %>%
  mutate(A_outlier_value = if_else(A_outlier == 0, NA_real_, A * A_outlier)
         , B_outlier_value = if_else(B_outlier == 0, NA_real_, B * B_outlier)
         , C_outlier_value = if_else(C_outlier == 0, NA_real_, C * C_outlier)
  )

A_outlier <- ggplot(df, aes(x=dt, y=A)) +
  geom_line(color=roam_palette[1]) +
  geom_point(aes(y=A_outlier_value), color='red', shape=20, size=2) +
  ggtitle("Series A") +
  xlab("") + ylab("") +
  theme_minimal()
ggsave("./fig/A_outlier.png", plot=A_outlier)

B_outlier <- ggplot(df, aes(x=dt, y=B)) +
  geom_line(color=roam_palette[1]) +
  geom_point(aes(y=B_outlier_value), color='red', shape=20, size=2) +
  annotate("text", x=as.POSIXct("2016-01-07 12:00:00"), y=-30, label="Potential\noutlier") +
  annotate("segment", x=as.POSIXct("2016-01-07 16:00:00"), xend=as.POSIXct("2016-01-07 16:00:00")
           , y=-26, yend=-12) +
  ggtitle("Series B") +
  xlab("") + ylab("") +
  theme_minimal()
ggsave("./fig/B_outlier.png", plot=B_outlier)

C_outlier <- ggplot(df, aes(x=dt, y=C)) +
  geom_line(color=roam_palette[1]) +
  geom_point(aes(y=C_outlier_value), color='red', shape=20, size=2) +
  annotate("text", x=as.POSIXct("2016-01-06"), y=-230, label="Extreme\noutliers") +
  annotate("segment", x=as.POSIXct("2016-01-04 13:00:00"), xend=as.POSIXct("2016-01-05 10:00:00")
           , y=-215, yend=-230) +
  annotate("segment", x=as.POSIXct("2016-01-03 15:00:00"), xend=as.POSIXct("2016-01-05 10:00:00")
           , y=-245, yend=-230) +
  annotate("text", x=as.POSIXct("2016-01-03 12:00:00"), y=300, label="Extreme\noutlier") +
  annotate("segment", x=as.POSIXct("2016-01-04 03:00:00"), xend=as.POSIXct("2016-01-05")
           , y=305, yend=305) +
  ggtitle("Series C") +
  xlab("") + ylab("") +
  theme_minimal()
ggsave("./fig/C_outlier.png", plot=C_outlier)
