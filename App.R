# app.R
# Trash Night Fit Predictor: cosine similarity + TMDb user score + genres + recs

library(shiny)
library(httr)
library(jsonlite)
library(text2vec)
library(dplyr)

# ← Insert your TMDb API key here ↓
tmdb_api_key <- "9d297056af68910628b79746ed1f9f86"

# 1) Training data (titles + your fit ratings)
train_titles  <- c(
  "Top Gun: Maverick",
  "Mortal Engines",
  "Der Eisbär",
  "Honey, I Shrunk the Kids",
  "Magic Mike",
  "Anaconda",
  "König der Herzen"
)
train_ratings <- c(1, 3, 2, 3, 3, 2, 2)

# 2) Helper to fetch TMDb features
get_info <- function(title, api_key) {
  search_url <- sprintf(
    "https://api.themoviedb.org/3/search/movie?api_key=%s&query=%s&language=en-US",
    api_key, URLencode(title)
  )
  res <- GET(search_url)
  if (res$status_code != 200) return(NULL)
  tmp <- fromJSON(rawToChar(res$content), simplifyVector = FALSE)
  if (length(tmp$results) == 0) return(NULL)
  mv <- tmp$results[[1]]
  
  det <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s?api_key=%s&language=en-US",
                mv$id, api_key))$content
  ), simplifyVector = FALSE)
  
  # first review
  revs <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s/reviews?api_key=%s&page=1",
                mv$id, api_key))$content
  ), simplifyVector = FALSE)$results
  review1 <- if (length(revs) > 0) revs[[1]]$content else ""
  
  # top 5 recommendations scores
  recs <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s/recommendations?api_key=%s&page=1",
                mv$id, api_key))$content
  ), simplifyVector = FALSE)$results
  rec_scores <- sapply(head(recs, 5), function(x) x$vote_average)
  if (length(rec_scores) < 5) rec_scores <- c(rec_scores, rep(NA, 5 - length(rec_scores)))
  
  # genres
  genres <- sapply(det$genres, `[[`, "name")
  
  list(
    user_score = det$vote_average %||% NA_real_,
    overview   = det$overview %||% "",
    review1    = review1,
    rec_scores = rec_scores,
    genres     = genres
  )
}

# 3) Fetch training features safely
# default fallback for get_info failures
default_info <- list(
  user_score = NA_real_,
  overview   = "",
  review1    = "",
  rec_scores = rep(NA_real_, 5),
  genres     = character(0)
)
# attempt fetch; wrap in tryCatch so app starts even if TMDb fails
t_info <- tryCatch({
  lst <- lapply(train_titles, get_info, api_key = tmdb_api_key)
  # replace any NULL entries
  lapply(lst, function(x) if (is.null(x)) default_info else x)
}, error = function(e) {
  warning("Training fetch failed: ", e$message)
  replicate(length(train_titles), default_info, simplify = FALSE)
})
# extract features
train_overview <- sapply(t_info, `[[`, "overview")
train_review1  <- sapply(t_info, `[[`, "review1")
X_rec          <- t(sapply(t_info, `[[`, "rec_scores"))
train_genres   <- lapply(t_info, `[[`, "genres")
train_score    <- sapply(t_info, `[[`, "user_score")

y_train        <- train_ratings
t_info <- lapply(train_titles, get_info, api_key = tmdb_api_key)
# Ensure no NULL entries (use safe defaults if fetch failed)
t_info <- lapply(t_info, function(x) {
  if (is.null(x)) {
    return(list(
      user_score = NA_real_,
      overview   = "",
      review1    = "",
      rec_scores = rep(NA_real_, 5),
      genres     = character(0)
    ))
  }
  x
})
train_overview <- sapply(t_info, `[[`, "overview")
train_review1  <- sapply(t_info, `[[`, "review1")
X_rec          <- t(sapply(t_info, `[[`, "rec_scores"))  # 7x5
train_genres   <- lapply(t_info, `[[`, "genres")
train_score    <- sapply(t_info, `[[`, "user_score")

y_train        <- train_ratings

# 4) Build TF–IDF on overview + review
texts_train <- paste(train_overview, train_review1, sep = " ")
it_train    <- itoken(texts_train, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
vocab       <- create_vocabulary(it_train) %>% prune_vocabulary(term_count_min = 1)
vectorizer  <- vocab_vectorizer(vocab)
dtm_train   <- create_dtm(it_train, vectorizer)
tfidf_model <- TfIdf$new()
X_text_train<- fit_transform(dtm_train, tfidf_model)
train_norms <- sqrt(rowSums(X_text_train ^ 2))

# 5) UI with German labels for weights
ui <- fluidPage(
  titlePanel("Trash Night Fit Predictor"),
  sidebarLayout(
    sidebarPanel(
      textInput("movie", "Movie Title:"),
      actionButton("go", "Predict"),
      sliderInput("w_text", "Gewichtung: Beschreibung", min = 0, max = 1, value = 0.4, step = 0.1),
      sliderInput("w_score", "Gewichtung: Bewertung", min = 0, max = 1, value = 0.2, step = 0.1),
      sliderInput("w_rec", "Gewichtung: Vorschläge für ähnliche Filme", min = 0, max = 1, value = 0.2, step = 0.1),
      sliderInput("w_gen", "Gewichtung: Genre", min = 0, max = 1, value = 0.2, step = 0.1)
    ),
    mainPanel(tableOutput("result"))
  )
)

# 6) Server: compute similarity-based fits and blend
server <- function(input, output) {
  output$result <- renderTable({
    req(input$go)
    info <- get_info(input$movie, tmdb_api_key)
    if (is.null(info)) {
      return(data.frame(title = input$movie, predicted_fit = NA_real_, stringsAsFactors = FALSE))
    }
    # Text similarity
    txt_new <- paste(info$overview, info$review1, sep = " ")
    it_new  <- itoken(txt_new, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
    dtm_new <- create_dtm(it_new, vectorizer)
    x_new   <- tfidf_model$transform(dtm_new)
    new_norm<- sqrt(sum(x_new^2))
    numer   <- as.numeric((X_text_train %*% t(x_new)))
    denom   <- train_norms * new_norm
    sims    <- numer / denom
    sims[is.na(sims)] <- 0
    text_fit <- if (sum(sims) > 0) sum(sims * train_ratings) / sum(sims) else mean(train_ratings)
    
    # Numeric score mapping
    score   <- info$user_score
    num_fit <- if (!is.na(score)) { f <- 3 - 2 * (score / 10); pmin(pmax(f, 1), 3) } else mean(train_ratings)
    
    # Rec scores mapping
    recs    <- info$rec_scores
    rec_fit <- if (length(recs) > 0) {
      fits <- 3 - 2 * (recs / 10)
      mean(fits, na.rm = TRUE)
    } else mean(train_ratings)
    
    # Genre Jaccard similarity
    new_genres <- info$genres
    jacc      <- sapply(train_genres, function(g) {
      u <- unique(c(g, new_genres)); if (length(u) == 0) return(0);
      length(intersect(g, new_genres)) / length(u)
    })
    genre_fit <- if (sum(jacc) > 0) sum(jacc * train_ratings) / sum(jacc) else mean(train_ratings)
    
    # Blend with German labels weights
    ws      <- c(input$w_text, input$w_score, input$w_rec, input$w_gen)
    names(ws)<- c("text","score","rec","genre")
    if (sum(ws) == 0) ws <- rep(1,4)
    pred    <- (ws["text"] * text_fit +
                  ws["score"] * num_fit +
                  ws["rec"]   * rec_fit +
                  ws["genre"] * genre_fit) / sum(ws)
    
    data.frame(
      title = input$movie,
      predicted_fit = round(pred, 2),
      stringsAsFactors = FALSE
    )
  })
}

shinyApp(ui, server)
