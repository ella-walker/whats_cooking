# Load libraries
library(tidyverse)
library(readr)
library(jsonlite)
library(stringr)
library(tidytext)  
library(purrr)  
library(recipes)
library(tidymodels)
library(embed)
library(textrecipes)
library(vroom)

# Read JSON files
trainSet <- read_file("~/Documents/STAT 348/whats_cooking/whats-cooking/train.json") |> 
  fromJSON()

testSet <- read_file("~/Documents/STAT 348/whats_cooking/whats-cooking/test.json") |> 
  fromJSON()

# trainSet has columns: id, cuisine, ingredients (list-column)

# Clean data
ingredient_counts <- trainSet %>%
  unnest(ingredients) %>%
  count(ingredients, sort = TRUE)
# view(ingredient_counts)
long <- trainSet %>% unnest(ingredients)


#### TF-IDF ####
trainSet <- trainSet |>
  count(id, cuisine, ingredients) |>
  bind_tf_idf(term=ingredients, document=id, n=n)
head(trainSet)

## DON'T UNNEST!!
cooking_recipe <- recipe(cuisine ~ ingredients, data = trainSet) |>
  step_mutate(ingredients = tokenlist(ingredients)) |>
  step_tokenfilter(ingredients, max_tokens=2500) |>
  step_tfidf(ingredients)


# MODEL (no tuning)
random_forest_mod <- rand_forest(
  mtry = 5,    
  min_n = 5,
  trees = 500
) |>
  set_engine("ranger") |>
  set_mode("classification")

# WORKFLOW
random_forest_wf <- workflow() |>
  add_recipe(cooking_recipe) |>
  add_model(random_forest_mod)

# FIT
random_forest_final <- random_forest_wf |>
  fit(data = trainSet)

# PREDICT
random_forest_preds <- predict(random_forest_final, new_data = testSet)

final_submission <- testSet |> 
  select(id) |>                    
  bind_cols(random_forest_preds) |> 
  rename(cuisine = .pred_class) |> 
  select(id, cuisine)

# SAVE
vroom_write(final_submission,
            file = "~/Documents/STAT 348/whats_cooking/random_forest_2.csv",
            delim = ",")










# Code from Danika
train_long <- trainSet %>% unnest(ingredients)
test_long  <- testSet %>% unnest(ingredients)

unique_ingredients <- train_long %>%
  mutate(ingredient = str_to_lower(ingredients)) %>%
  
  # Count cuisines per ingredient
  distinct(cuisine, ingredient) %>%
  count(ingredient, name = "num_cuisines") %>% 
  filter(num_cuisines == 1) %>%     # keep only ingredients appearing in exactly one cuisine
  
  # Join back to get the cuisine info
  inner_join(
    train_long %>%
      mutate(ingredient = str_to_lower(ingredients)) %>%
      distinct(cuisine, ingredient),
    by = "ingredient"
  ) %>%
  
  # Add the number of unique recipe IDs containing this ingredient
  left_join(
    train_long %>%
      mutate(ingredient = str_to_lower(ingredients)) %>%
      group_by(ingredient) %>%
      summarise(num_recipes = n_distinct(id), .groups = "drop"),
    by = "ingredient"
  ) %>%
  
  arrange(cuisine, ingredient)


train_prepped <- trainSet %>% 
  mutate(ingredient_text = map_chr(ingredients, ~ paste(tolower(.x), collapse = " ")))

test_prepped <- testSet %>% 
  mutate(ingredient_text = map_chr(ingredients, ~ paste(tolower(.x), collapse = " ")))


cooking_recipe <- recipe(cuisine ~ ingredient_text, data = train_prepped) %>%
  step_tokenize(ingredient_text) %>%
  step_stopwords(ingredient_text) %>%
  step_tokenfilter(ingredient_text, max_tokens = 1000) %>%
  step_tf(ingredient_text)

# -------------------------------
# Multinomial Logistic Regression Model
# -------------------------------
logistic_mod <- multinom_reg(penalty = 1e-4) %>%  # optional small regularization
  set_engine("glmnet") %>%
  set_mode("classification")

# -------------------------------
# Workflow
# -------------------------------
logistic_wf <- workflow() %>%
  add_recipe(cooking_recipe) %>%
  add_model(logistic_mod)

# -------------------------------
# Fit Final Model
# -------------------------------
logistic_final <- logistic_wf %>%
  fit(data = train_prepped)

# -------------------------------
# Predict on Test Data
# -------------------------------
logistic_preds <- predict(logistic_final, new_data = test_prepped)

# -------------------------------
# Create Kaggle Submission
# -------------------------------
final_submission <- test_prepped %>%
  select(id) %>%
  bind_cols(logistic_preds) %>%
  rename(cuisine = .pred_class)

# Save CSV
vroom_write(
  final_submission,
  file = "~/Documents/STAT 348/whats_cooking/logistic.csv",
  delim = ","
)
