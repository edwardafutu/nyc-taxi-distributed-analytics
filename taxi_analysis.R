# ---------------------------------------------------------------------
install.packages("arrow")   # if not already installed
library(arrow)        # read_parquet(): reads the Parquet file into an R data frame
library(sparklyr)     # R <-> Spark bridge: copy_to(), ml_*, sdf_*
library(dplyr)        # cleaning verbs, run in R this time (not pushed to Spark)
library(lubridate)    # hour() and wday() for time based features
library(tidyr)        # reshaping for tidy tables later
library(ggplot2)
#remove.packages("cli")
#install.packages ("cli")
#remove.packages("tidyr")
#install.packages("tidyr")  # diagnostic and correlation plots
library(tidymodels)   # parsnip, rsample, yardstick, broom in one call
library(corrr)        # correlate() workflow used in Part C
library(rsample)
#install.packages("yardstick")
library(yardstick)
tidymodels_prefer()



# ---------------------------------------------------------------------
# 12. LOCAL SPARK CONNECTION
# ---------------------------------------------------------------------
sc <- spark_connect(master = "local")


# ---------------------------------------------------------------------
# 13. LOAD THE PARQUET INTO R (not Spark yet)
# ---------------------------------------------------------------------
taxi_r_raw <- read_parquet("yellow_tripdata_2026-01.parquet")

dim(taxi_r_raw)                     # expect 3724889 x 20
str(taxi_r_raw, list.len = 20)      # column names and R types

# INSPECT: first rows
head(taxi_r_raw, 5)


# ---------------------------------------------------------------------
# 14a. UNDERSTAND WHAT NEEDS CLEANING (diagnostic)
# ---------------------------------------------------------------------
# Counts that justify each filter below (all computed in R).
taxi_r_raw %>%
  summarise(
    n_total    = n(),
    pay_type0  = sum(payment_type == 0,  na.rm = TRUE),   # expect 1,088,058
    fare_neg   = sum(fare_amount   < 0,  na.rm = TRUE),   # expect    39,463
    total_le0  = sum(total_amount <= 0,  na.rm = TRUE),   # expect    40,417
    dist_le0   = sum(trip_distance <= 0, na.rm = TRUE),   # expect   125,738
    dist_gt100 = sum(trip_distance > 100,na.rm = TRUE),
    pax0       = sum(passenger_count == 0, na.rm = TRUE)  # expect    14,787
  )


# ---------------------------------------------------------------------
# 14b. CLEANING + FEATURE ENGINEERING (all in R)
# ---------------------------------------------------------------------
taxi_r <- taxi_r_raw %>%
  # normalise code column types (arrow may read some as integer64)
  mutate(
    passenger_count = as.integer(passenger_count),
    RatecodeID      = as.integer(RatecodeID),
    payment_type    = as.integer(payment_type)
  ) %>%
  
  # keep the two well defined payment methods
  # (also drops the payment_type = 0 block that holds the missing values)
  filter(payment_type %in% c(1, 2)) %>%
  
  # time based derived features
  mutate(
    trip_duration_min = as.numeric(difftime(tpep_dropoff_datetime,
                                            tpep_pickup_datetime,
                                            units = "mins")),
    pickup_hour       = hour(tpep_pickup_datetime),
    pickup_dow        = wday(tpep_pickup_datetime)   # 1 = Sun ... 7 = Sat
  ) %>%
  mutate(
    avg_speed_mph = if_else(trip_duration_min > 0,
                            trip_distance / (trip_duration_min / 60),
                            NA_real_)
  ) %>%
  
  # remove impossible or extreme records surfaced during profiling
  filter(
    fare_amount       > 0,
    total_amount      > 0,
    trip_distance     > 0,  trip_distance     <= 100,
    passenger_count   >= 1, passenger_count   <= 5,
    trip_duration_min > 0,  trip_duration_min <= 180,
    avg_speed_mph     > 0,  avg_speed_mph     <= 80
  ) %>%
  
  # engineered labels and ratios
  mutate(
    is_weekend    = if_else(pickup_dow %in% c(1, 7), 1L, 0L),
    is_airport    = if_else(coalesce(Airport_fee, 0) > 0 |
                              RatecodeID %in% c(2, 3), 1L, 0L),
    payment_label = if_else(payment_type == 1, "card", "cash"),
    tip_pct       = if_else(fare_amount > 0, tip_amount / fare_amount, 0)
  ) %>%
  
  # code variables as factors so they behave as categoricals
  mutate(
    VendorID   = as.factor(VendorID),
    RatecodeID = as.factor(RatecodeID)
  ) %>%
  
  # keep only the modelling columns and drop any residual missing rows
  select(
    total_amount, fare_amount, tip_amount, tip_pct,
    trip_distance, trip_duration_min, avg_speed_mph,
    passenger_count, pickup_hour, is_weekend, is_airport,
    VendorID, RatecodeID, payment_label
  ) %>%
  na.omit()


# ---------------------------------------------------------------------
# 14c. CONFIRM THE CLEANED R DATA FRAME
# ---------------------------------------------------------------------
# Before / after row counts
data.frame(
  stage = c("raw", "cleaned", "removed"),
  rows  = c(nrow(taxi_r_raw), nrow(taxi_r),
            nrow(taxi_r_raw) - nrow(taxi_r))
)                                   

glimpse(taxi_r)                     # engineered columns and types

# INSPECT: class balance for classification (Part G)
count(taxi_r, payment_label)       
count(taxi_r, is_airport)           


# ---------------------------------------------------------------------
# 14d. COPY THE CLEANED DATA INTO SPARK
# ---------------------------------------------------------------------
# copy_to() serialises the cleaned R frame and registers it as a Spark
# DataFrame. For ~2.5M rows this takes a short while and uses driver
# memory; that is expected. (Faster alternative for very large frames:
# arrow::write_parquet(taxi_r, "taxi_clean.parquet") then
# spark_read_parquet(sc, "taxi", "taxi_clean.parquet").)
taxi <- copy_to(sc, taxi_r, name = "taxi", overwrite = TRUE)

class(taxi)                         
sdf_dim(taxi)                      
tbl_cache(sc, "taxi")               # cache in Spark memory for reuse in C-G


# ---------------------------------------------------------------------
# 15. REPRODUCIBILITY + SPARK VERSION
# ---------------------------------------------------------------------
set.seed(757)                       # reused in sdf_random_split() in Part D
spark_version(sc)                   # documents the Spark version used

# spark_disconnect(sc)              # when finished for the session


# =====================================================================
# Part C: Correlation Analysis
# Requires taxi_r (R data frame) and taxi (Spark table) from Part B
# =====================================================================

# ---------------------------------------------------------------------
# Numerical columns used by all three methods
# ---------------------------------------------------------------------
num_vars <- taxi_r %>%
  select(total_amount, fare_amount, tip_amount, tip_pct, trip_distance,
         trip_duration_min, avg_speed_mph, passenger_count, pickup_hour,
         is_weekend, is_airport)


# ---------------------------------------------------------------------
# 16. Base R correlation matrix
# ---------------------------------------------------------------------
cor_base <- round(cor(num_vars),1)


# ---------------------------------------------------------------------
# 17. Spark correlation matrix with ml_corr()
# ---------------------------------------------------------------------
cor_spark <- round(ml_corr(
  taxi,
  columns = c("total_amount", "fare_amount", "tip_amount", "tip_pct",
              "trip_distance", "trip_duration_min", "avg_speed_mph",
              "passenger_count", "pickup_hour", "is_weekend", "is_airport"),
  method = "pearson"
),1)

# If your sparklyr version drops the row/column names, use instead:
# round(as.matrix(cor_spark), 1)


# ---------------------------------------------------------------------
# 18. Correlation matrix with the corrr package
# ---------------------------------------------------------------------
cor_corrr <- num_vars %>%
  correlate(use = "pairwise.complete.obs", method = "pearson")
fashion(cor_corrr, decimals = 1)


# ---------------------------------------------------------------------
# 19. Correlation heatmap
# ---------------------------------------------------------------------
cor_long <- as.data.frame(cor_base) %>%
  tibble::rownames_to_column("var1") %>%
  pivot_longer(-var1, names_to = "var2", values_to = "r")

ggplot(cor_long, aes(var1, var2, fill = r)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = round(r, 1)), size = 3) +
  scale_fill_gradient2(low = "#B2182B", mid = "white", high = "#2166AC",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation heatmap of numerical variables",
       x = NULL, y = NULL, fill = "Pearson r") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


###SECTION D
##training-test split
splits    <- sdf_random_split(
  taxi, training = 0.8, test = 0.2,
  seed = 757)
taxi_train <- splits$training
taxi_test  <- splits$test

#checking the number of rows of each partition
sdf_nrow(taxi_test)
sdf_nrow(taxi_train)

##22. Training test split in base R
set.seed(757)
taxi_split <- initial_split(taxi_r)
taxi_train_r <- training(taxi_split)
taxi_test_r <- testing(taxi_split)

##checking their dimensions
dim(taxi_test_r)
dim(taxi_train_r)

##23.Loading the r-based training and test sets into Spark
Spark_test <- copy_to(sc, taxi_test_r)
Spark_train <- copy_to(sc, taxi_train_r)

class(Spark_test)
class(Spark_train)


#E. Regression modelling in Spark
?ml_linear_regression
SparkMod1 <- ml_linear_regression(taxi, formula = tip_amount ~ trip_distance)
summary(SparkMod1)

#2. Standardized predictor
taxi <- taxi %>%
  mutate(trip_distance_z =
           (trip_distance - mean(trip_distance, na.rm = TRUE))/sd(trip_distance))

glimpse(taxi)

#same model with standardized predictor
SparkMod1z <- ml_linear_regression(taxi, formula = tip_amount ~ trip_distance_z)
summary(SparkMod1z)

#3. model output using tidy
tidy(SparkMod1)
tidy(SparkMod1z)

##4. Generate predictions on the training set and calculate residuals and RMSE
?ml_predict()
SparkMod1pred <- ml_predict(SparkMod1, taxi_train)
glimpse(SparkMod1pred)

SparkMod1pred <- SparkMod1pred %>% 
  mutate(res = tip_amount - prediction,
         Std_res = res/SparkMod1$summary$root_mean_squared_error)

#5. Residual diagnostic plots
#collecting a sample of the data into R
plot_df <- SparkMod1pred %>%
  sdf_sample(fraction = 0.02, replacement = FALSE, seed = 757) %>%
  select(prediction, res, Std_res) %>%
  collect()

#Residual vs fitted plot
ggplot(plot_df, aes(x = prediction, y = res)) +
  geom_point(alpha = 0.15, colour = "#2166AC") +
  geom_hline(yintercept = 0, colour = "#B2182B", linewidth = 0.7) +
  geom_smooth(se = FALSE, colour = "black", linewidth = 0.6) +
  labs(title = "Residuals vs. Fitted",
       x = "Fitted values (predicted tip_amount)",
       y = "Residuals") +
  theme_minimal(base_size = 11)

#standardized residuals vs fitted values
ggplot(plot_df, aes(x = prediction, y = Std_res)) +
  geom_point(alpha = 0.15, colour = "#2166AC") +
  geom_hline(yintercept = 0, colour = "#B2182B", linewidth = 0.7) +
  geom_hline(yintercept = c(-2, 2), linetype = "dashed",
             colour = "grey40") +
  labs(title = "Standardised Residuals vs. Fitted",
       x = "Fitted values (predicted tip_amount)",
       y = "Standardised residuals") +
  theme_minimal(base_size = 11)

#6. multiple linear regression using ml_linear_regression
SparkMLRmod1 <- ml_linear_regression(taxi, tip_amount ~ 
                                       trip_distance+fare_amount+payment_label)
tidy(SparkMLRmod1)
summary(SparkMLRmod1)

#7. fitting the models using parsnip
#SLR model using parsnip
SLRmod1_parsnip <- fit(linear_reg(engine = "spark"),
                   formula = tip_amount~trip_distance , data = taxi) 
tidy(SLRmod1_parsnip)

MLRmod1_parsnip <- fit(linear_reg(engine = "spark"),
                       formula = tip_amount ~ 
                         trip_distance+fare_amount+payment_label, data = taxi) 
tidy(MLRmod1_parsnip)

# =====================================================================
# Part G: Classification Modelling in Spark
# Outcome variable used: payment_label ("card" or "cash")
# We reuse the SAME cleaned dataset (taxi_r) rather than a new file, as
# allowed by the brief ("may be the same as the primary dataset if a
# suitable outcome variable exists"). payment_label already exists as a
# raw, non-engineered field (built directly from payment_type in Part
# B), so this is a genuinely natural binary outcome: "will a passenger
# pay by card or by cash, given the characteristics of the trip?"
#
# NOTE ON LEAKAGE - IMPORTANT: NYC taxi meters only capture TIPS made
# through the card reader. Cash tips are handed to the driver directly
# and are never logged, so tip_amount (and tip_pct, which is built
# from it) is ~$0 for almost every cash trip but >$0 for most card
# trips.This means tip_amount (and tip_pct) basically gives away the
# answer instead of genuinely predicting it. total_amount
# is contaminated too, since it is literally
# fare_amount + tip_amount + fees, so it inherits the same leak.
# For this reason tip_amount, tip_pct, and total_amount are all
# EXCLUDED from the predictor list below. fare_amount (the base fare,
# fixed before any tip) is kept, since it carries no such leak.
# =====================================================================


# ---------------------------------------------------------------------
# 11. PREPARE THE CLASSIFICATION DATASET
# ---------------------------------------------------------------------
#  a) using already-cleaned taxi_r data frame (Part B)
#  b) make sure payment_label is a proper factor (it is currently a
#     plain character column: "card" / "cash")
#  c) keep only the columns we want as predictors + the outcome -
#     deliberately DROPPING tip_amount, tip_pct and total_amount (see
#     the leakage note above), and dropping RatecodeID/payment_type
#     since RatecodeID is not relevant here and payment_type is just
#     the numeric version of the outcome itself
#  d) drop any rows that still have missing values, then check the result


class_r <- taxi_r %>%
  mutate(
    payment_label = as.factor(payment_label)      # character -> factor ("card"/"cash")
  ) %>%
  select(
    payment_label,                             # <-- outcome (what we predict)
    trip_distance, trip_duration_min, avg_speed_mph,   # numeric predictors
    fare_amount,                                       # base fare only (no tip/leak)
    passenger_count, pickup_hour, is_weekend,          # numeric predictors
    is_airport,                                        # numeric (0/1) predictor
    VendorID                                           # categorical predictor
  ) %>%
  na.omit()                                    # drop any row with a missing value

# CHECK 1: how many rows had missing values removed (should be 0, taxi_r was
# already cleaned in Part B, so this is really just a safety check)
nrow(taxi_r) - nrow(class_r)

# CHECK 2: confirm there are now no missing values left anywhere
sum(is.na(class_r))                            # expect 0

# CHECK 3: look at the class balance of the outcome variable
# (this matters a lot later when we discuss model performance - "card"
# is the much bigger group here, so this outcome is imbalanced too,
# just less extremely than is_airport was)
count(class_r, payment_label)

# CHECK 4:Verifying structure of outcome variable
str(class_r$payment_label)

# ---------------------------------------------------------------------
# 12. TRAIN/TEST SPLIT FOR CLASSIFICATION (in R, then copied to Spark)
# ---------------------------------------------------------------------
set.seed(757)                                  # same seed used throughout, for reproducibility
class_split   <- initial_split(class_r)        # do the 75/25 split
class_train_r <- training(class_split)         # pull out the 75% training rows
class_test_r  <- testing(class_split)          # pull out the 25% test rows

# CHECK: dimensions of the R-based train/test sets
dim(class_train_r)
dim(class_test_r)

# Loading to R
class_train <- copy_to(sc, class_train_r, name = "class_train", overwrite = TRUE)
class_test  <- copy_to(sc, class_test_r,  name = "class_test",  overwrite = TRUE)

# CHECK: confirm both pieces really are Spark DataFrames, and report their
# dimensions inside Spark (should match the R dimensions above)
class(class_train)
class(class_test)
sdf_dim(class_train)
sdf_dim(class_test)

# ---------------------------------------------------------------------
# 13. DEFINE THREE PARSNIP MODEL SPECIFICATIONS (Spark engine)
# ---------------------------------------------------------------------
model_specs <- list(
  logistic = logistic_reg(engine = "spark", mode = "classification"),
  boosted  = boost_tree(engine  = "spark", mode = "classification"),
  forest   = rand_forest(engine = "spark", mode = "classification")
)

model_specs                                    # print to show all three specs


# ---------------------------------------------------------------------
# 14. FIT ALL THREE MODELS - SINGLE PREDICTOR (Spark training data)
# ---------------------------------------------------------------------
## Remember to show this code in report 
single_formula <- payment_label ~ fare_amount

fits_single <- lapply(model_specs, function(spec) {
  fit(spec, formula = single_formula, data = class_train)
})

# CHECK: names carried over from model_specs, confirming we have 3 fitted
# models (logistic, boosted, forest)
names(fits_single)

 ---------------------------------------------------------------------
# 15. PREDICT ON THE SPARK TEST SET, COLLECT INTO R
# ---------------------------------------------------------------------
# predict() run once per fitted model (again via lapply). Because
# new_data = class_test is still a Spark DataFrame, the predictions are
# computed inside Spark, and predict() returns one small Spark DataFrame
# of predicted classes per model.
preds_single <- lapply(fits_single, predict, new_data = class_test)

# Pull the TRUE labels for the test set down into R once (same for all
# three models, since they all used the same class_test set)
truth_single <- class_test %>%
  select(payment_label) %>%
  collect() %>%
  pull(payment_label) %>%
  as.factor()

# For each model: collect its predictions into R, attach the true labels
# as a column, and make sure the predicted class is a factor (yardstick
# needs both truth and estimate to be factors with the same levels)
results_single <- lapply(preds_single, function(p) {
  p %>%
    collect() %>%                              # Spark -> R
    mutate(.pred_class = factor(.pred_class, levels = levels(truth_single))) %>%
    mutate(payment_label = truth_single)       # attach ground truth
})

# --------------------------------------------------------------------
# 15. PREDICT ON THE SPARK TEST SET, COLLECT INTO R
# ---------------------------------------------------------------------

preds_single <- lapply(fits_single, predict, new_data = class_test)

# Pull the TRUE labels for the test set down into R once (same for all
# three models, since they all used the same class_test set)
truth_single <- class_test %>%
  select(payment_label) %>%
  collect() %>%
  pull(payment_label) %>%
  as.factor()

# For each model: collect its predictions into R, attach the true labels
# as a column, and make sure the predicted class is a factor (yardstick
# needs both truth and estimate to be factors with the same levels)
results_single <- lapply(preds_single, function(p) {
  p %>%
    collect() %>%                              # Spark -> R
    mutate(pred_class = factor(pred_class, levels = levels(truth_single))) %>%
    mutate(payment_label = truth_single)       # attach ground truth
})

# ---------------------------------------------------------------------
# 16. ACCURACY FOR ALL THREE MODELS (single predictor)
# ---------------------------------------------------------------------
acc_single <- lapply(names(results_single), function(nm) {
  accuracy(results_single[[nm]], truth = payment_label, estimate = pred_class) %>%
    mutate(model = nm, predictors = "single (fare_amount)")
}) %>%
  bind_rows() %>%
  arrange(desc(.estimate))

acc_single                                     # print: accuracy, ranked best -> worst

# ---------------------------------------------------------------------
# 17. RE-FIT ALL THREE MODELS - ALL PREDICTORS
# ---------------------------------------------------------------------
# Same process as steps 14-16, but the formula now uses "." to mean
# "every other column in class_train except payment_label". Remember,
# tip_amount/tip_pct/total_amount were already excluded back in step 11
# precisely so that "." here cannot quietly leak the answer.
all_formula <- payment_label ~ .

fits_all <- lapply(model_specs, function(spec) {
  fit(spec, formula = all_formula, data = class_train)
})

preds_all <- lapply(fits_all, predict, new_data = class_test)

# truth_single already holds the correct test-set labels and can be reused,
# since class_test itself has not changed between step 15 and here.
results_all <- lapply(preds_all, function(p) {
  p %>%
    collect() %>%
    mutate(pred_class = factor(pred_class, levels = levels(truth_single))) %>%
    mutate(payment_label = truth_single)
})

acc_all <- lapply(names(results_all), function(nm) {
  accuracy(results_all[[nm]], truth = payment_label, estimate = pred_class) %>%
    mutate(model = nm, predictors = "all predictors")
}) %>%
  bind_rows() %>%
  arrange(desc(.estimate))

acc_all                                        # print: accuracy, ranked best -> worst

# COMPARISON TABLE: single-predictor vs. all-predictors accuracy, side by
# side for each of the three models.
acc_comparison <- bind_rows(acc_single, acc_all) %>%
  select(model, predictors, accuracy = .estimate) %>%
  pivot_wider(names_from = predictors, values_from = accuracy) %>%
  arrange(desc(`all predictors`))

acc_comparison                                 # final comparison table

# ---------------------------------------------------------------------
# 18. FEATURE IMPORTANCE FOR THE BOOSTED TREE AND RANDOM FOREST
# ---------------------------------------------------------------------
# We only do this for the "all
# predictors" models (fits_all), since a single-predictor model has
# nothing interesting to rank.
tidy(fits_all$boosted)                         # boosted tree importances
tidy(fits_all$forest)                          # random forest importances

# Sorted, easier-to-read versions (highest importance first)
tidy(fits_all$boosted) %>% arrange(desc(importance))
tidy(fits_all$forest)  %>% arrange(desc(importance))

# ---------------------------------------------------------------------
# 19. DISCUSSION 
# ---------------------------------------------------------------------
# (a) Best model: acc_comparison - typically the tree ensembles
#     (boosted tree / random forest) beat plain logistic regression once
#     all predictors are used, because they can capture non-linear
#     relationships and interactions (e.g. airport trips and long,
#     high-fare trips behaving differently to short local trips) that a
#     linear logistic model cannot.
# (b) Complexity vs. interpretability: logistic regression is the most
#     interpretable (one coefficient per predictor, easy to explain to a
#     non-technical audience - e.g. "each extra dollar of fare shifts
#     the log-odds of paying by card by X") but the least flexible/accuracte.
#     Random forest and boosted trees are more accurate but but harder to explain 
#      - tidy() importances tell us WHICH predictors matter
#     but not HOW, the way a logistic regression coefficient's sign and
#     size would.
# (c) Overfitting: payment_label is imbalanced, so a
#     model can look accurate just by predicting "card" almost every
#     time. k-fold cross-validation (e.g. vfold_cv() from rsample)
#     instead of a single train/test split would give a more honest
#     estimate of how well each model generalises, and metrics beyond
#     accuracy (e.g. sensitivity, specificity, or ROC-AUC from
#     yardstick) would better reflect performance on the rarer "cash"
#     class.

