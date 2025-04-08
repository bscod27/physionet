Compute.Missingness <- function(df) {
  #' Computes the proportion of missingness per column of a dataframe
  #' Arguments
  #'  x: input dataframe
  return(sapply(df, function(i) sum(is.na(i)))/nrow(df))
}

P.Value <- function(x, ...) {
  #' Computes a p-value from a chi-squared test of independence based on a table1
  #' Arguments
  #'  x: input table
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  p <- chisq.test(table(y, g))$p.value 
  p <- ifelse(p < 0.01, '<0.01', sprintf("%.2f", round(p, 2)))
}
  
Pivot.Wide <- function(df) {
  #' Pivots a dataframe with discretized time points into wide format
  #' Arguments
  #'  df: input dataframe
  out <- df %>% 
    pivot_wider(
      id_cols = PATIENT_ID,
      names_from = c(Parameter, Period), 
      values_from = Value
    )
  return(out)
}
