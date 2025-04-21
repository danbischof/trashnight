# app.R

library(shiny)
library(httr)
library(jsonlite)
library(text2vec)
library(dplyr)

# safe “or‐NULL” operator
`%||%` <- function(a, b) if (!is.null(a)) a else b

# 1) Your TMDb key (hard‑coded for now)
tmdb_api_key <- "9d297056af68910628b79746ed1f9f86"

# 2) Training titles + ratings
train_titles  <- c(
  "Top Gun: Maverick",
  "Mortal Engines",
  "Der Eisbär",
  "Honey, I Shrunk the Kids",
  "Magic Mike",
  "9360-anaconda",
  "König der Herzen",
  "Ghost Rider",
  "Titanic",
  "Pulp Fiction"
)
train_ratings <- c(1, 3, 2, 3, 3, 2, 2, 1, 3, 3)

# 3) Fetch features for one movie
get_info <- function(title, api_key) {
  sr <- GET(sprintf(
    "https://api.themoviedb.org/3/search/movie?api_key=%s&query=%s&language=en-US",
    api_key, URLencode(title)
  ))
  if (sr$status_code != 200) return(NULL)
  tmp <- fromJSON(rawToChar(sr$content), simplifyVector=FALSE)
  if (length(tmp$results)==0) return(NULL)
  mv <- tmp$results[[1]]
  
  det <- fromJSON(rawToChar(
    GET(sprintf(
      "https://api.themoviedb.org/3/movie/%s?api_key=%s&language=en-US",
      mv$id, api_key
    ))$content
  ), simplifyVector=FALSE)
  
  revs <- fromJSON(rawToChar(
    GET(sprintf(
      "https://api.themoviedb.org/3/movie/%s/reviews?api_key=%s&page=1",
      mv$id, api_key
    ))$content
  ), simplifyVector=FALSE)$results
  review1 <- if (length(revs)>0) revs[[1]]$content else ""
  
  recs <- fromJSON(rawToChar(
    GET(sprintf(
      "https://api.themoviedb.org/3/movie/%s/recommendations?api_key=%s&page=1",
      mv$id, api_key
    ))$content
  ), simplifyVector=FALSE)$results
  rec_scores <- sapply(head(recs,5), function(x) x$vote_average)
  if (length(rec_scores)<5) rec_scores <- c(rec_scores, rep(NA,5-length(rec_scores)))
  
  genres <- sapply(det$genres, `[[`, "name")
  
  list(
    user_score = det$vote_average %||% NA_real_,
    overview   = det$overview   %||% "",
    review1    = review1,
    rec_scores = rec_scores,
    genres     = genres
  )
}

# 4) PRE‑COMPUTE training embeddings
train_info  <- lapply(train_titles, get_info, api_key=tmdb_api_key)
train_over  <- sapply(train_info, `[[`, "overview")
train_rev   <- sapply(train_info, `[[`, "review1")
texts_train <- paste(train_over, train_rev, sep=" ")
it_train    <- itoken(texts_train, tolower, word_tokenizer, progressbar=FALSE)
vocab       <- create_vocabulary(it_train) %>% prune_vocabulary(term_count_min=1)
vectorizer  <- vocab_vectorizer(vocab)
dtm_train   <- create_dtm(it_train, vectorizer)
tfidf_model <- TfIdf$new()
X_text_train<- fit_transform(dtm_train, tfidf_model)
train_norms <- if (is.matrix(X_text_train) && ncol(X_text_train)>0) {
  sqrt(rowSums(X_text_train^2))
} else rep(0, length(train_titles))

# 5) UI
ui <- fluidPage(
  titlePanel("Trash Night Fit Predictor"),
  sidebarLayout(
    sidebarPanel(
      textInput("movie","Movie title:"),
      actionButton("go","Predict"),
      sliderInput("w_text",  "Gewichtung: Beschreibung",   0,1,0.4,step=0.1),
      sliderInput("w_score","Gewichtung: Bewertung",  0,1,0.2,step=0.1),
      sliderInput("w_rec",  "Gewichtung: Vorschläge für ähnliche Filme ",   0,1,0.2,step=0.1),
      sliderInput("w_gen",  "Gewichtung: Genre ",  0,1,0.2,step=0.1)
    ),
    mainPanel(tableOutput("result"))
  )
)

# 6) Server: compute fits + blend
server <- function(input, output) {
  output$result <- renderTable({
    req(input$go)
    info <- get_info(input$movie, tmdb_api_key)
    if (is.null(info)) return(data.frame(title=input$movie, predicted_fit=NA_real_))
    
    # Text similarity via cosine
    txt_new <- paste(info$overview, info$review1, sep=" ")
    it_new  <- itoken(txt_new, tolower, word_tokenizer, progressbar=FALSE)
    dtm_new <- create_dtm(it_new, vectorizer)
    # full 1 x vocab_size matrix
    # ————————————
    #  a) Embed the new text and coerce to a base matrix
    x_new_sparse <- tfidf_model$transform(dtm_new)
    # force it into a standard dense matrix (1 x M)
    x_new_mat    <- as.matrix(x_new_sparse)
    
    #  b) Now do the cosine‐similarity safely
    X_base   <- as.matrix(X_text_train)   # your 7×M training matrix
    numer    <- X_base %*% t(x_new_mat)   # 7×M %*% M×1 → 7×1
    new_norm <- sqrt(sum(x_new_mat^2))    # L2 norm of new row
    denom    <- train_norms * new_norm    # vector length 7
    sims     <- ifelse(denom > 0, numer[,1] / denom, 0)
    text_fit <- if (sum(sims) > 0) sum(sims * train_ratings) / sum(sims) else mean(train_ratings)
    # ————————————
    
    
    # Numeric score fitting
    score   <- info$user_score
    num_fit <- if (!is.na(score)) {
      f <- 3 - 2 * (score/10)
      pmin(pmax(f,1),3)
    } else mean(train_ratings)
    
    # Recommendation fitting
    recs    <- info$rec_scores
    rec_fit <- if (any(!is.na(recs))) {
      f <- 3 - 2 * (recs/10)
      mean(f, na.rm=TRUE)
    } else mean(train_ratings)
    
    # Genre Jaccard similarity
    train_genres <- lapply(train_info, `[[`, "genres")
    new_g        <- info$genres
    jacc         <- sapply(train_genres, function(g) {
      u <- unique(c(g, new_g))
      if (length(u) == 0) return(0)
      length(intersect(g, new_g)) / length(u)
    })
    gen_fit <- if (sum(jacc) > 0) sum(jacc * train_ratings) / sum(jacc) else mean(train_ratings)
    
    # Blend weights
    ws <- c(input$w_text, input$w_score, input$w_rec, input$w_gen)
    if (sum(ws) == 0) ws <- rep(1, 4)
    pred <- (ws[1] * text_fit + ws[2] * num_fit + ws[3] * rec_fit + ws[4] * gen_fit) / sum(ws)
    
    data.frame(title = input$movie,
               predicted_fit = round(pred, 2),
               stringsAsFactors = FALSE)
  })
}

shinyApp(ui, server)
