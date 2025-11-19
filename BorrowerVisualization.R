library(tidyverse)
library(readxl)

borrower <- read_excel("C:/Users/prohi/PycharmProjects/POC/ML/data/output/borrowers_with_clusters_vae_2.xlsx")
print(borrower)
glimpse(borrower)

borrower_df <-
  borrower |>
  mutate(cluster_n = paste0("cluster", "_", borrower$risk_group))

borrower_df |>
  summarize(
    avg_credit_utl = mean(credit_utilization) * 100,
    avg_debt_to_income = mean(debt_to_income) * 100,
    avg_outstanding_loan_amt = mean(loan_amount_outstanding) * 100,
    n = n(),
    .by = cluster_n
  )

cred_util_dist <-
ggplot(borrower_df, aes(x = cluster_n, y = credit_utilization)) +
  geom_boxplot(
    fill = "lightgreen",
    color = "black",
    outlier.size = 3,
    outlier.color = "red"
  ) +
  geom_jitter(
    width = 0.3,
    size = 1,
    color = "orange",
    alpha = 0.5
  ) +
  labs(
    title = "Box plot",
    subtitle = "distribution of credit utilization by cluster",
    x = "credit risk group",
    y = "credit utilization"
  )

dti_dist <-
  ggplot(borrower_df, aes(x = cluster_n, y = debt_to_income)) +
  geom_boxplot(
    fill = "lightgreen",
    color = "black",
    outlier.size = 3,
    outlier.color = "red"
  ) +
  geom_jitter(
    width = 0.3,
    size = 1,
    color = "orange",
    alpha = 0.5
  ) +
  labs(
    title = "Box plot",
    subtitle = "distribution of debt-to-income by cluster",
    x = "credit risk group",
    y = "debt-to-income"
  )

out_amt_dist <-
  ggplot(borrower_df, aes(x = cluster_n, y = loan_amount_outstanding)) +
  geom_boxplot(
    fill = "lightgreen",
    color = "black",
    outlier.size = 3,
    outlier.color = "red"
  ) +
  geom_jitter(
    width = 0.3,
    size = 1,
    color = "orange",
    alpha = 0.5
  ) +
  labs(
    title = "Box plot",
    subtitle = "distribution of outstanding loan amount by cluster",
    x = "credit risk group",
    y = "outstanding loan amount"
  )

income_vol_dist <-
  ggplot(borrower_df, aes(x = cluster_n, y = income_volatility)) +
  geom_boxplot(
    fill = "lightgreen",
    color = "black",
    outlier.size = 3,
    outlier.color = "red"
  ) +
  geom_jitter(
    width = 0.3,
    size = 1,
    color = "orange",
    alpha = 0.5
  ) +
  labs(
    title = "Box plot",
    subtitle = "distribution of income volatility by cluster",
    x = "credit risk group",
    y = "income volatility"
  )


ggsave(
  "C:/Users/prohi/PycharmProjects/POC/ML/data/output/credit_utilization_dist.jpeg",
  plot = cred_util_dist,
  width = 6,
  height = 4,
  dpi = 300
)

ggsave(
  "C:/Users/prohi/PycharmProjects/POC/ML/data/output/dti_dist.jpeg",
  plot = dti_dist,
  width = 6,
  height = 4,
  dpi = 300
)

ggsave(
  "C:/Users/prohi/PycharmProjects/POC/ML/data/output/out_amt_dist.jpeg",
  plot = out_amt_dist,
  width = 6,
  height = 4,
  dpi = 300
)

ggsave(
  "C:/Users/prohi/PycharmProjects/POC/ML/data/output/income_vol_dist.jpeg",
  plot = income_vol_dist,
  width = 6,
  height = 4,
  dpi = 300
)
