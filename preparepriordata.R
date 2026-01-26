library(dplyr)
library(tidyr)
library(stringr)

exp <- 11

setwd("~/Documents/GitHub/reliable_info_bias")

data <- read.csv("data/DATA_Exp11_Unaware_Blue.csv")


data <- data %>%
  separate(Sample_Reliability, into = paste0("proba_", 1:6), sep = "\\s+") %>%
  mutate(across(starts_with("proba_"), as.numeric)) %>%
  mutate(color = str_extract_all(Sample_Color, "blue|red")) %>%
  unnest_wider(color, names_sep = "_")

write.csv(data, "data_priorbelief_unaware_blue_exp11.csv", row.names = FALSE)
save(data, file = "data_priorbelief_unaware_blue_exp11.rdata")