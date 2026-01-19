library(dplyr)
library(tidyr)
library(stringr)

exp <- 12

setwd("~/Documents/GitHub/reliable_info_bias")

data <- read.csv("data/DATA_Exp12_Aware_Red.csv")

## Exp12 has 6 samples
data <- data %>%
  separate(Sample_Reliability, into = paste0("proba_", 1:6), sep = "\\s+") %>%
  mutate(across(starts_with("proba_"), as.numeric)) %>%
  mutate(color = str_extract_all(Sample_Color, "blue|red")) %>%
  unnest_wider(color, names_sep = "_")

write.csv(data, "data_priorbelief_aware_red_exp12.csv", row.names = FALSE)
save(data, file = "data_priorbelief_aware_red_exp12.rdata")