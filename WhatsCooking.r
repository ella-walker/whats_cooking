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
ingredient_counts <- train %>%
  unnest(ingredients) %>%
  count(ingredients, sort = TRUE)
# view(ingredient_counts)
long <- train %>% unnest(ingredients)

##### CODE FROM JULIA ######
all_cuisines <- unique(long$cuisine)

overall_counts <- long %>%
  count(ingredients, name = "count_overall")

enrichment_all <- map_dfr(all_cuisines, function(cui) {
  
  target_counts <- long %>%
    filter(cuisine == cui) %>%
    count(ingredients, name = "count_in_cuisine")
  
  left_join(target_counts, overall_counts, by = "ingredients") %>%
    mutate(
      cuisine = cui,
      ratio = count_in_cuisine / count_overall,
      weighted_count_ratio = count_in_cuisine * ratio
    ) %>%
    arrange(desc(weighted_count_ratio))
})

enrichment_deduplicate <- enrichment_all %>%
  group_by(ingredients) %>%
  slice_max(weighted_count_ratio, n = 1, with_ties = FALSE) %>%
  ungroup()

top500 <- enrichment_deduplicate %>%
  group_by(cuisine) %>%
  arrange(desc(weighted_count_ratio)) %>%
  mutate(global_rank = row_number()) %>%
  slice_head(n = 500) %>%
  ungroup()

test_long <- test %>% unnest(ingredients)
# Join test ingredients to the ranked predictors
match_table <- test_long %>%
  inner_join(top500, by = "ingredients")

best_match <- match_table %>%
  arrange(id, global_rank, cuisine) %>%   # break ties deterministically
  group_by(id) %>%
  slice(1) %>% 
  ungroup() %>%
  select(id, predicted_cuisine = cuisine)

all_cuisines <- unique(train$cuisine)

missing_ids <- setdiff(test$id, best_match$id)

random_assign <- tibble(
  id = missing_ids,
  predicted_cuisine = sample(all_cuisines, length(missing_ids), replace = TRUE)
)

final_predictions <- bind_rows(best_match, random_assign)

submission <- final_predictions %>%
  arrange(id) %>%
  select(id, cuisine = predicted_cuisine)

write_csv(submission,
          "~/Documents/STAT 348/whats_cooking/preds.csv")


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
  mtry = 5,        # pick a fixed value
  min_n = 5,       # pick a fixed value
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
