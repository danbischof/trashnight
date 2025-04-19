
# Assumes the following objects are available:
# tmdb_api_key, vectorizer, tfidf, cvmod, lam

# Simplified search: prefer German search first, fallback to English
search_movie_tmdb <- function(movie_title, api_key) {
  # Helper to pull search results in a given language, returns tibble or NULL
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
  
  # 1) Try German search
  res_de <- fetch_search("de-DE")
  if (!is.null(res_de)) {
    best <- res_de[1, ]  # take first German result
  } else {
    # 2) Fallback to English search
    res_en <- fetch_search("en-US")
    if (is.null(res_en)) return(NULL)
    best <- res_en[1, ]
  }
  
  # Fetch details in English for the chosen ID
  det_url <- sprintf(
    "https://api.themoviedb.org/3/movie/%s?api_key=%s&language=en-US",
    best$id, api_key
  )
  det_res <- GET(det_url)
  if (det_res$status_code != 200) return(NULL)
  det <- fromJSON(rawToChar(det_res$content))
  
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

# Prediction function remains unchanged
detect_pipe <- function() TRUE # placeholder
predict_fit_for_titles <- function(titles, api_key, vectorizer, tfidf, cvmod, lam) {
  meta_list <- lapply(titles, function(t) {
    md <- search_movie_tmdb(t, api_key)
    if (is.null(md)) return(data.frame(title = t, description = NA_character_, predicted_fit = NA_real_))
    md
  })
  meta_df <- bind_rows(meta_list)
  it_new <- itoken(
    meta_df$description,
    preprocessor = tolower,
    tokenizer    = word_tokenizer,
    progressbar  = FALSE
  )
  dtm_new <- create_dtm(it_new, vectorizer)
  dtm_tfidf_new <- tfidf$transform(dtm_new)
  X_new <- as.matrix(dtm_tfidf_new)
  fits <- rep(NA_real_, nrow(meta_df))
  valid <- !is.na(meta_df$description)
  if (any(valid)) {
    fits[valid] <- as.numeric(
      predict(cvmod, newx = X_new[valid, , drop = FALSE], s = lam)
    )
  }
  meta_df %>%
    mutate(predicted_fit = fits) %>%
    select(title, predicted_fit)
}

# Shiny UI & Server
ui <- fluidPage(
  titlePanel("Trash Night Fit Predictor"),
  sidebarLayout(
    sidebarPanel(
      textAreaInput(
        "titles", "Movie Titles (one per line):", rows = 5,
        placeholder = "z.B. Stirb langsam 3"
      ),
      actionButton("go", "Vorhersage anzeigen")
    ),
    mainPanel(
      tableOutput("result_table")
    )
  )
)

server <- function(input, output, session) {
  predictions <- eventReactive(input$go, {
    req(input$titles)
    titles <- strsplit(input$titles, "\\n")[[1]] %>% trimws()
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
  output$result_table <- renderTable({
    predictions()
  }, rownames = FALSE)
}

shinyApp(ui, server)
