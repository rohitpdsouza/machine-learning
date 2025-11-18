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
    subtitle = "distribution of credit_utilization",
    x = "cluster_n",
    y = "credit_utilization"
  )


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
    subtitle = "distribution of debt_to_income",
    x = "cluster_n",
    y = "debt_to_income"
  )

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
    subtitle = "distribution of loan_amount_oustanding_by_cluster",
    x = "cluster_n",
    y = "outstanding_loan_amount"
  )