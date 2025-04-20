# app.R

library(shiny)
library(httr)
library(jsonlite)
library(text2vec)
library(dplyr)

# Define safe 'or' operator
`%||%` <- function(a, b) if (!is.null(a)) a else b

# ← your TMDb key ↓
tmdb_api_key <- "9d297056af68910628b79746ed1f9f86"

# 1) Training titles + your 1–3 ratings
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

# 2) Fetch all features for one movie
get_info <- function(title, api_key) {
  sr <- GET(sprintf(
    "https://api.themoviedb.org/3/search/movie?api_key=%s&query=%s&language=en-US",
    api_key, URLencode(title)
  ))
  if (sr$status_code != 200) return(NULL)
  tmp <- fromJSON(rawToChar(sr$content), simplifyVector=FALSE)
  if (length(tmp$results)==0) return(NULL)
  mv <- tmp$results[[1]]
  # details
  det <- fromJSON(rawToChar(
    GET(sprintf(
      "https://api.themoviedb.org/3/movie/%s?api_key=%s&language=en-US",
      mv$id, api_key
    ))$content
  ), simplifyVector=FALSE)
  # first review
  revs <- fromJSON(rawToChar(
    GET(sprintf(
      "https://api.themoviedb.org/3/movie/%s/reviews?api_key=%s&page=1",
      mv$id, api_key
    ))$content
  ), simplifyVector=FALSE)$results
  review1 <- if (length(revs)>0) revs[[1]]$content else ""
  # top‑5 rec scores
  recs <- fromJSON(rawToChar(
    GET(sprintf(
      "https://api.themoviedb.org/3/movie/%s/recommendations?api_key=%s&page=1",
      mv$id, api_key
    ))$content
  ), simplifyVector=FALSE)$results
  rec_scores <- sapply(head(recs,5), function(x) x$vote_average)
  if (length(rec_scores)<5) rec_scores <- c(rec_scores, rep(NA,5-length(rec_scores)))
  # genres
  genres <- sapply(det$genres, `[[`, "name")
  list(
    user_score = det$vote_average %||% NA_real_,
    overview   = det$overview   %||% "",
    review1    = review1,
    rec_scores = rec_scores,
    genres     = genres
  )
}

# 3) PRE‑COMPUTE training data once
train_info   <- lapply(train_titles, get_info, api_key=tmdb_api_key)
train_over   <- sapply(train_info, `[[`, "overview")
train_rev    <- sapply(train_info, `[[`, "review1")
train_rec    <- t(sapply(train_info, `[[`, "rec_scores"))  # 7x5
train_genres <- lapply(train_info, `[[`, "genres")

# 4) Build TF–IDF on overview + review1
texts_train   <- paste(train_over, train_rev, sep=" ")
it_train      <- itoken(texts_train, tolower, word_tokenizer, progressbar=FALSE)
vocab         <- create_vocabulary(it_train) %>% prune_vocabulary(term_count_min=1)
vectorizer    <- vocab_vectorizer(vocab)
dtm_train     <- create_dtm(it_train, vectorizer)
tfidf_model   <- TfIdf$new()
X_text_train  <- fit_transform(dtm_train, tfidf_model)
# safe norms
if (is.matrix(X_text_train) && ncol(X_text_train)>0) {
  train_norms <- sqrt(rowSums(X_text_train^2))
} else {
  train_norms <- rep(0, length(train_titles))
}

# 5) UI: weight sliders
ui <- fluidPage(
  titlePanel("Trash Night Fit Predictor"),
  sidebarLayout(
    sidebarPanel(
      textInput("movie", "Movie title:"),
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
    # text sim
    txt_new <- paste(info$overview, info$review1, sep=" ")
    it_new  <- itoken(txt_new, tolower, word_tokenizer, progressbar=FALSE)
    dtm_new <- create_dtm(it_new, vectorizer)
    x_new   <- tfidf_model$transform(dtm_new)
    if (is.null(dim(x_new))) x_new <- matrix(x_new, nrow=1)
    new_norm <- sqrt(sum(x_new^2))
    numer    <- as.numeric(X_text_train %*% t(x_new))
    denom    <- train_norms * new_norm
    sims     <- ifelse(denom>0, numer/denom, 0)
    text_fit <- if (sum(sims)>0) sum(sims*train_ratings)/sum(sims) else mean(train_ratings)
    # numeric score
    score   <- info$user_score
    num_fit <- if (!is.na(score)) {
      f <- 3 - 2*(score/10); pmin(pmax(f,1),3)
    } else mean(train_ratings)
    # recs
    recs    <- info$rec_scores
    rec_fit <- if (any(!is.na(recs))) {
      f <- 3 - 2*(recs/10); mean(f, na.rm=TRUE)
    } else mean(train_ratings)
    # genre jaccard
    new_g   <- info$genres
    jacc    <- sapply(train_genres, function(g) {
      u <- unique(c(g,new_g)); if (length(u)==0) 0 else length(intersect(g,new_g))/length(u)
    })
    gen_fit <- if (sum(jacc)>0) sum(jacc*train_ratings)/sum(jacc) else mean(train_ratings)
    # blend
    ws <- c(input$w_text, input$w_score, input$w_rec, input$w_gen)
    if (sum(ws)==0) ws <- rep(1,4)
    pred <- (ws[1]*text_fit + ws[2]*num_fit + ws[3]*rec_fit + ws[4]*gen_fit)/sum(ws)
    data.frame(title=input$movie, predicted_fit=round(pred,2), stringsAsFactors=FALSE)
  })
}

shinyApp(ui, server)
