# app.R
# Trash Night Fit Predictor: combining TMDb user score, genres, overview, first review, rec scores

library(shiny)
library(httr)
library(jsonlite)
library(text2vec)
library(dplyr)

# ← Insert your TMDb API key here ↓
tmdb_api_key <- Sys.getenv("TMDB_API_KEY")  # expects TMDB_API_KEY set in environment

# 1) Training data: titles + your fit ratings
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

# 2) Helper to fetch all features for one movie
get_info <- function(title, api_key) {
  # search
  sr <- GET(sprintf(
    "https://api.themoviedb.org/3/search/movie?api_key=%s&query=%s&language=en-US",
    api_key, URLencode(title)
  ))
  if (sr$status_code != 200) return(NULL)
  tmp <- fromJSON(rawToChar(sr$content), simplifyVector = FALSE)
  if (length(tmp$results) == 0) return(NULL)
  mv <- tmp$results[[1]]
  
  # details
  det <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s?api_key=%s&language=en-US",
                mv$id, api_key))$content
  ), simplifyVector = FALSE)
  
  # first review
  revs <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s/reviews?api_key=%s&page=1",
                mv$id, api_key))$content
  ), simplifyVector = FALSE)$results
  review1 <- if (length(revs)>0) revs[[1]]$content else ""
  
  # top 5 recommendation scores
  recs <- fromJSON(rawToChar(
    GET(sprintf("https://api.themoviedb.org/3/movie/%s/recommendations?api_key=%s&page=1",
                mv$id, api_key))$content
  ), simplifyVector = FALSE)$results
  rec_scores <- sapply(head(recs,5), function(x) x$vote_average)
  if (length(rec_scores) < 5) rec_scores <- c(rec_scores, rep(NA,5-length(rec_scores)))
  
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

# 3) Pre-fetch training features
train_info <- lapply(train_titles, get_info, api_key = tmdb_api_key)
# extract lists
train_overview <- sapply(train_info, `[[`, "overview")
train_review1  <- sapply(train_info, `[[`, "review1")
train_rec      <- t(sapply(train_info, `[[`, "rec_scores"))  # 7x5 matrix
train_genres   <- lapply(train_info, `[[`, "genres")
train_score    <- sapply(train_info, `[[`, "user_score")

# 4) Build combined TF–IDF on overview + review1
texts_train <- paste(train_overview, train_review1, sep = " ")
it_train    <- itoken(
  texts_train,
  preprocessor = tolower,
  tokenizer    = word_tokenizer,
  progressbar  = FALSE
)
vocab       <- create_vocabulary(it_train) %>% prune_vocabulary(term_count_min = 1)
vectorizer  <- vocab_vectorizer(vocab)
dtm_train   <- create_dtm(it_train, vectorizer)
tfidf_model <- TfIdf$new()
X_text_train<- fit_transform(dtm_train, tfidf_model)
train_norms <- sqrt(rowSums(X_text_train ^ 2))

# 5) Unique genres list for Jaccard
all_genres <- sort(unique(unlist(train_genres)))

# 6) UI: inputs and weight sliders
ui <- fluidPage(
  titlePanel("Trash Night Fit Predictor"),
  sidebarLayout(
    sidebarPanel(
      textInput("movie", "Movie title:"),
      actionButton("go", "Predict"),
      sliderInput("w_text",  "Gewichtung: Beschreibung",   0,1,0.4,step=0.1),
      sliderInput("w_score","Gewichtung: Bewertung",  0,1,0.2,step=0.1),
      sliderInput("w_rec",  "Gewichtung: Vorschläge für ähnliche Filme ",   0,1,0.2,step=0.1),
      sliderInput("w_gen",  "Gewichtung: Genre ",  0,1,0.2,step=0.1)
    ),
    mainPanel(tableOutput("result"))
  )
)

# 7) Server: compute each component and blend
server <- function(input, output) {
  output$result <- renderTable({
    req(input$go)
    info <- get_info(input$movie, tmdb_api_key)
    if (is.null(info)) {
      return(data.frame(title = input$movie, predicted_fit = NA_real_))
    }
    # Text similarity
    txt_new <- paste(info$overview, info$review1, sep = " ")
    it_new  <- itoken(txt_new, tolower, word_tokenizer, FALSE)
    dtm_new <- create_dtm(it_new, vectorizer)
    x_new   <- tfidf_model$transform(dtm_new)
    new_norm<- sqrt(sum(x_new^2))
    numer   <- as.numeric((X_text_train %*% t(x_new)))
    denom   <- train_norms * new_norm
    sims    <- numer / denom
    sims[is.na(sims)] <- 0
    text_fit<- if (sum(sims)>0) sum(sims * train_ratings)/sum(sims) else mean(train_ratings)
    
    # Numeric score mapping
    score   <- info$user_score
    num_fit <- if (!is.na(score)) { f <- 3 - 2*(score/10); pmin(pmax(f,1),3) } else mean(train_ratings)
    
    # Rec scores mapping
    recs    <- info$rec_scores
    recs_fit<- if (length(recs)>0) {
      rfits <- 3 - 2*(recs/10)
      mean(rfits, na.rm=TRUE)
    } else mean(train_ratings)
    
    # Genre similarity (Jaccard)
    new_genres <- info$genres
    jacc       <- sapply(train_genres, function(g) {
      len <- length(unique(c(g, new_genres)))
      if (len==0) return(0)
      length(intersect(g, new_genres))/len
    })
    genre_fit <- if (sum(jacc)>0) sum(jacc * train_ratings)/sum(jacc) else mean(train_ratings)
    
    # Blend with weights (normalized)
    ws   <- c(input$w_text, input$w_score, input$w_rec, input$w_gen)
    names(ws) <- c('text','score','rec','genre')
    ws_sum <- sum(ws)
    if (ws_sum==0) ws_sum <- 1
    pred <- (ws['text']*text_fit +
               ws['score']*num_fit +
               ws['rec']*recs_fit +
               ws['genre']*genre_fit) / ws_sum
    
    data.frame(
      title = input$movie,
      predicted_fit = round(pred,2),
      stringsAsFactors = FALSE
    )
  })
}

shinyApp(ui, server)