# app.R

# Load required packages
library(shiny)
library(dplyr)
library(magrittr)
library(httr)
library(jsonlite)
library(text2vec)
library(stringdist)
library(glmnet)

# Source model artifacts (must define: tmdb_api_key, vectorizer, tfidf, cvmod, lam)
source("Classifier.R")

# Function: search a movie on TMDb (German then English)
search_movie_tmdb <- function(movie_title, api_key) {
  fetch_search <- function(lang) {
    url <- sprintf(
      "https://api.themoviedb.org/3/search/movie?api_key=%s&language=%s&query=%s",
      api_key, lang, URLencode(movie_title)
    )
    res <- GET(url)
    if (res$status_code != 200) return(NULL)
    dat <- fromJSON(rawToChar(res$content))
    if (length(dat$results) == 0) return(NULL)
    as_tibble(dat$results)
  }
  # try German search
  res_de <- fetch_search("de-DE")
  if (!is.null(res_de)) {
    best <- res_de[1, ]
  } else {
    # fallback to English
    res_en <- fetch_search("en-US")
    if (is.null(res_en)) return(NULL)
    best <- res_en[1, ]
  }
  det_url <- sprintf(
    "https://api.themoviedb.org/3/movie/%s?api_key=%s&language=en-US",
    best$id, api_key
  )
  det_res <- GET(det_url)
  if (det_res$status_code != 200) return(NULL)
  det <- fromJSON(rawToChar(det_res$content))
  genres <- NA_character_
  if (!is.null(det$genres) && length(det$genres) > 0) {
    if (is.data.frame(det$genres)) {
      genres <- paste(det$genres$name, collapse = ", ")
    } else {
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

# Predict fit for a vector of titles
predict_fit_for_titles <- function(titles, api_key, vectorizer, tfidf, cvmod, lam) {
  # Fetch metadata for each title
  meta_list <- lapply(titles, function(t) {
    md <- search_movie_tmdb(t, api_key)
    if (is.null(md)) {
      return(data.frame(title = t, description = NA_character_, predicted_fit = NA_real_))
    }
    md
  })
  meta_df <- bind_rows(meta_list)
  # Embed descriptions via TF-IDF
  it_new <- itoken(meta_df$description,
                   preprocessor = tolower,
                   tokenizer    = word_tokenizer,
                   progressbar  = FALSE)
  dtm_new <- create_dtm(it_new, vectorizer)
  dtm_tfidf_new <- tfidf$transform(dtm_new)
  X_new <- as.matrix(dtm_tfidf_new)
  # Predict
  fits <- rep(NA_real_, nrow(meta_df))
  valid <- !is.na(meta_df$description)
  if (any(valid)) {
    fits[valid] <- predict(cvmod, newx = X_new[valid, , drop = FALSE], s = lam)
  }
  meta_df %>%
    mutate(predicted_fit = as.numeric(fits)) %>%
    select(title, predicted_fit)
}

# Shiny UI
ui <- fluidPage(
  titlePanel("Trash Night Fit Predictor"),
  sidebarLayout(
    sidebarPanel(
      textAreaInput(
        inputId = "titles",
        label   = "Movie Titles (one per line):",
        rows    = 5,
        placeholder = "Bsp. Stirb langsam: Jetzt erst recht"
      ),
      actionButton(inputId = "go", label = "Predict Fit")
    ),
    mainPanel(
      tableOutput("result_table")
    )
  )
)

# Shiny Server
server <- function(input, output, session) {
  results <- eventReactive(input$go, {
    req(input$titles)
    titles <- strsplit(input$titles, "\n")[[1]] %>% trimws()
    titles <- titles[titles != ""]
    predict_fit_for_titles(
      titles     = titles,
      api_key    = tmdb_api_key,
      vectorizer = vectorizer,
      tfidf      = tfidf,
      cvmod      = cvmod,
      lam        = lam
    )
  })
  output$result_table <- renderTable({ results() }, rownames = FALSE)
}

shinyApp(ui, server)
