# install.packages(c("httr", "jsonlite", "dplyr", "text", "glmnet"))
library(httr)
library(jsonlite)
library(dplyr)
library(text)
library(glmnet)
library(text2vec)
library(shiny)
library(stringdist)

# ← YOUR TMDb API KEY HERE ↓
tmdb_api_key <- "9d297056af68910628b79746ed1f9f86"

# 1) Helper: search + fetch details for one title
search_movie_tmdb <- function(movie_title, api_key) {
  # Search endpoint
  url <- sprintf(
    "https://api.themoviedb.org/3/search/movie?api_key=%s&query=%s",
    api_key, URLencode(movie_title)
  )
  res <- GET(url)
  if (res$status_code != 200) return(NULL)
  dat <- fromJSON(rawToChar(res$content))
  if (length(dat$results) == 0) return(NULL)
  mv <- dat$results[1, ]  # first row
  
  # Full details endpoint
  det <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s?api_key=%s",
                mv$id, api_key))$content
  ))
  
  # Safe genres extraction
  genres <- NA_character_
  if (!is.null(det$genres) && length(det$genres) > 0) {
    if (is.data.frame(det$genres)) {
      genres <- paste(det$genres$name, collapse = ", ")
    } else if (is.list(det$genres)) {
      genres <- paste(sapply(det$genres, `[[`, "name"), collapse = ", ")
    }
  }
  
  data.frame(
    title       = det$title,
    description = det$overview,
    rating_tmdb = det$vote_average,
    genres      = genres,
    stringsAsFactors = FALSE
  )
}

# 2) YOUR “Trash Night” movies + your 1–3 ratings
trash_night_movies <- data.frame(
  title  = c(
    "Top Gun: Maverick", 
    "Mortal Engines", 
    "Der Eisbär", 
    "Honey, I shrunk the kids", 
    "Magic Mike", 
    "Anaconda", 
    "König der Herzen"
  ),
  rating = c(1, 3, 2, 3, 3, 2, 2),
  stringsAsFactors = FALSE
)

# 3) Fetch metadata for your rated movies, preserving the order and rating
trash_data_list <- lapply(seq_len(nrow(trash_night_movies)), function(i) {
  title_i  <- trash_night_movies$title[i]
  rating_i <- trash_night_movies$rating[i]      # ← use 'rating' here
  md       <- search_movie_tmdb(title_i, tmdb_api_key)
  if (is.null(md)) return(NULL)
  md$your_rating   <- rating_i
  md$is_candidate  <- FALSE
  md
})

# Drop failed lookups
trash_data_list   <- trash_data_list[!sapply(trash_data_list, is.null)]
trash_movie_data  <- bind_rows(trash_data_list)

#sanity checks: 
print(nrow(trash_movie_data))          
print(trash_movie_data$title)          
print(trash_movie_data$your_rating)    
print(table(trash_movie_data$your_rating))

# bind into one data.frame
trash_movie_data <- bind_rows(trash_data_list)

# remove any NULLs (not found)
trash_data_list <- trash_data_list[!sapply(trash_data_list, is.null)]
trash_movie_data <- bind_rows(trash_data_list)

# align your_rating by title
trash_movie_data <- trash_movie_data %>%
  left_join(trash_night_movies, by = "title")

# 4) Build a candidate pool via TMDb Discover
# 4) Build a candidate pool of ~1000 movies via TMDb Discover

# (a) Discover function remains the same
get_discover <- function(page, api_key) {
  url <- sprintf(
    paste0(
      "https://api.themoviedb.org/3/discover/movie",
      "?api_key=%s",
      "&sort_by=vote_average.asc",
      "&vote_count.gte=100",       # ≥100 votes
      "&vote_average.gte=2",       # avoid the absolute worst
      "&vote_average.lte=7",       # avoid the very best
      "&page=%d"
    ),
    api_key, page
  )
  dat <- fromJSON(rawToChar(GET(url)$content))
  dat$results
}

# (b) Grab pages 1 through 50 → 50 * 20 = 1000 movies
pages <- 1:500
candidate_briefs <- bind_rows(
  lapply(pages, function(p) {
    Sys.sleep(0.2)               # throttle requests
    get_discover(p, tmdb_api_key)
  })
)

# (c) Truncate to exactly 1000 (in case some pages are shorter)
candidate_briefs <- head(candidate_briefs, 10000)


# — now we only need one extra API call to translate genre_ids → names —

# (d) Fetch the TMDb genre mapping once
genre_url   <- sprintf("https://api.themoviedb.org/3/genre/movie/list?api_key=%s", tmdb_api_key)
genre_list  <- fromJSON(rawToChar(GET(genre_url)$content))$genres
# genre_list is a data.frame with columns id and name

# (e) Build your candidate data.frame directly from the discover results
candidate_movie_data <- candidate_briefs %>%
  mutate(
    # TMDb fields:
    title       = title,
    description = overview,
    rating_tmdb = vote_average,
    # map genre_ids → comma‑separated names
    genres      = sapply(
      genre_ids,
      function(ids) {
        paste( genre_list$name[ genre_list$id %in% ids ], collapse = ", " )
      }
    )
  ) %>%
  select(title, description, rating_tmdb, genres)

# 5) Combine for modeling
trash_movie_data$is_candidate <- FALSE
trash_movie_data$pred_rating <- trash_movie_data$your_rating
trash_movie_data$your_rating <- trash_movie_data$your_rating

candidate_movie_data$is_candidate <- TRUE
candidate_movie_data$your_rating <- NA
candidate_movie_data$pred_rating <- NA

all_movies <- bind_rows(trash_movie_data, candidate_movie_data)

# --- STEP 6 (TF–IDF embedding instead of BERT) ---
it <- itoken(
  all_movies$description,
  preprocessor = tolower,
  tokenizer    = word_tokenizer,
  progressbar  = FALSE
)

# build a vocabulary (only keep terms that appear ≥5 times)
vocab <- create_vocabulary(it) %>%
  prune_vocabulary(term_count_min = 5)

vectorizer <- vocab_vectorizer(vocab)

# create document-term matrix
dtm <- create_dtm(it, vectorizer)

# apply TF–IDF
tfidf <- TfIdf$new()
dtm_tfidf <- fit_transform(dtm, tfidf)

# convert to a dense matrix for glmnet
X <- as.matrix(dtm_tfidf)

# Inspect your training labels
train_idx <- which(!is.na(all_movies$your_rating))
y_train   <- all_movies$your_rating[train_idx]
print(y_train)
print(table(y_train))

# --- STEP 7 (train regression) ---
train_idx <- which(!is.na(all_movies$your_rating))
X_train   <- X[train_idx, ]
y_train   <- all_movies$your_rating[train_idx]

cvmod <- cv.glmnet(
  x      = X_train,
  y      = y_train,
  family = "gaussian", 
  alpha  = 0      # ridge
)
lam <- cvmod$lambda.min


# --- STEP 8 (predict on all movies) ---
all_movies$pred_rating <- as.numeric(
  predict(cvmod, newx = X, s = lam)
)

# then rank candidates as before:
results <- all_movies %>%
  filter(is_candidate) %>%
  arrange(pred_rating) %>%
  select(title, rating_tmdb, genres, predicted_fit = pred_rating)

print(head(results, 20))



# test proposals:

predict_fit_for_titles <- function(titles, api_key, vectorizer, tfidf, cvmod, lam) {
  # Hilfsfunktion: TMDb‑Suche + Details (nutzt deinen search_movie_tmdb)
  fetch_one <- function(title) {
    md <- search_movie_tmdb(title, api_key)
    if (is.null(md)) {
      return(data.frame(
        title = title,
        description = NA_character_,
        warning = "not found",
        stringsAsFactors = FALSE
      ))
    }
    md
  }
  
  # 1) Metadaten holen
  meta_list <- lapply(titles, fetch_one)
  meta_df   <- bind_rows(meta_list)
  
  # 2) TF–IDF einbetten
  library(text2vec)
  it_new <- itoken(
    meta_df$description,
    preprocessor = tolower,
    tokenizer    = word_tokenizer,
    progressbar  = FALSE
  )
  dtm_new <- create_dtm(it_new, vectorizer)
  dtm_tfidf_new <- tfidf$transform(dtm_new)
  X_new <- as.matrix(dtm_tfidf_new)
  
  # 3) Vorhersage
  fits <- rep(NA_real_, nrow(meta_df))
  valid <- !is.na(meta_df$description)
  if (any(valid)) {
    fits[valid] <- as.numeric(
      predict(cvmod, newx = X_new[valid, , drop = FALSE], s = lam)
    )
  }
  
  # 4) Ergebnis zurückgeben
  meta_df %>%
    mutate(
      predicted_fit = fits
    ) %>%
    select(title, predicted_fit)
}

# ------------- Beispielaufruf -------------
my_suggestions <- c(
  "Die harder 1", 
  "Snakes on a Plane", 
  "The Emoji Movie"
)

predictions <- predict_fit_for_titles(
  titles     = my_suggestions,
  api_key    = tmdb_api_key,
  vectorizer = vectorizer,
  tfidf      = tfidf,
  cvmod      = cvmod,
  lam        = lam
)

print(predictions)

