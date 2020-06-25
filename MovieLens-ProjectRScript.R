#### MovieLens Project - Generating movie rating and RMSE ####
#### Author: Pritesh Contractor ####

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# split the edx set into 2 parts the training set and the test set 

#set.seed(1, sample.kind = "Rounding")
trainset_index <- createDataPartition(y=edx$rating, times=1, p=0.1, list=FALSE)
train_set <- edx[-trainset_index,]
temp <- edx[trainset_index,]

# ensure that userid and movieid in test set are also in train set 

test_set <- temp %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by="userId")

# adding rows removed from the test set back into the train set 

removed <- anti_join(temp,test_set)
train_set <- rbind(train_set,removed)

rm(trainset_index, temp, removed)

##### Part 2: Data Exploration #####

# structure of dataset

str(edx)

# rows and columns within the edx dataset

dim(edx)

head(edx)

# genre extraction 

edx %>% group_by(genres) %>% summarise(n=n()) %>% head()

# extracting number of genres in each movie listed with the table 

tibble(count=str_count(edx$genres, fixed("|")), genres=edx$genres) %>% 
  group_by(count, genres) %>% 
  summarise(n=n()) %>% 
  arrange(-count) %>% 
  head()

# date extraction

library(lubridate)
tibble('Initial Date'=date(as_datetime(min(edx$timestamp), origin="1970-01-01")),'Final Date'=date(as_datetime(max(edx$timestamp), origin="1970-01-01"))) %>%
  mutate(Period=duration(max(edx$timestamp)-min(edx$timestamp)))

# plotting number of ratings vs rating distribution year 

if(!require(ggthemes)) 
  install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) 
  install.packages("scales", repos = "http://cran.us.r-project.org")
edx %>% mutate(year=year(as_datetime(timestamp, origin="1970-01-01"))) %>%
  ggplot(aes(x=year)) +
  geom_histogram(color="white") +
  ggtitle("Rating Distribution per year") +
  xlab("Year") +
  ylab("Number of Ratings") +
  scale_y_continuous(labels=comma) +
  theme_grey()

# extracting table with more ratings 

edx %>% mutate(date=date(as_datetime(timestamp, origin="1970-01-01"))) %>%
  group_by(date,title) %>%
  summarise(count=n()) %>%
  arrange(-count) %>%
  head(10)

# extracting ratings - counting number of each ratings 

edx %>% group_by(rating) %>% summarize(n=n())

# extracting ratings within edx data set 

edx %>% group_by(rating) %>%
  summarise(count=n()) %>%
  ggplot(aes(x=rating,y=count)) +
  geom_line() + 
  geom_point() + 
  scale_y_log10(breaks=trans_breaks("log10", function(x) 10^x),labels=trans_format("log10", math_format(10^.x))) +
  ggtitle("Rating Distribution", subtitle = "Higher ratings are prevalent.") +
  xlab("Rating") +
  ylab("Count") +
  theme_grey()

# distribution of movies 

edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color="white") +
  scale_x_log10() +
  ggtitle("Distribution of Movies", subtitle = "The distribution is almost symmetric.") +
  xlab("Number of Ratings") +
  ylab("Number of Movies") +
  theme_grey()

# Extraction of users 

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  arrange(n) %>%
  head()

# user distribution 

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color="white") +
  scale_x_log10() +
  ggtitle("Distribution of Users", subtitle = "The distribution is right skewed.") +
  xlab("Number of Ratings") +
  ylab("Numnber of Users") +
  scale_y_continuous(labels = comma) +
  theme_grey()

##### Part 3: Data Cleaning #####

# extracting estimated rating for movie and user only - many predictors increases the model complexity and requires more computer resources 

train_set <- train_set %>% select(userId, movieId, rating, title)
test_set <- test_set %>% select(userId, movieId, rating, title)

##### Part 4: Modelling #####

# Evaluating Model Functions 


# defining Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, pred_ratings){
  sqrt(mean((true_ratings-pred_ratings)^2))
}

# random prediction model - rating distribution 

set.seed(4321, sample.kind = "Rounding")

# produce probability of each rating 

p <- function(x,y) mean(y==x)
rating <- seq(0.5,5,0.5)

# estimating probability of each rating with Monte Carlo Simulation 

B <- 10^3
M <- replicate(B,{
  s <- sample(train_set$rating,100,replace = TRUE)
  sapply(rating,p,y= s)
})

prob <- sapply(1:nrow(M), function(x) mean(M[x,]))

# predicting random ratings distribution 

pred_random <- sample(rating,size=nrow(test_set),
                      replace=TRUE, prob=prob)

# Table with the error results for random prediction model 

result <- tibble(Method="Goal of Project", RMSE=0.8649)
result <- bind_rows(result,tibble(Method="Random Prediction Model",
                                  RMSE=RMSE(test_set$rating, pred_random)))

print.data.frame(result)
  
# Linear Model Prediction for rating distribution 

# Mean of observed values 

mu <- mean(train_set$rating)

# Updating the error table 

result <- bind_rows(result,
                    tibble(Method="Mean",
                           RMSE=RMSE(test_set$rating, mu)))

# intermediate RMSE results

print.data.frame(result)

# including movie effect 

bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i=mean(rating-mu))
head(bi)

# plotting movie effect distribution 

bi %>% ggplot(aes(x=b_i)) +
  geom_histogram(bins=10, col=I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie Effect") +
  ylab("Count") +
  scale_y_continuous(labels=comma)+
  theme_grey()

# prediction of rating distribution with mean and bi

pred_bi <- mu + test_set %>%
  left_join(bi, by="movieId") %>%
  .$b_i

# calculating RMSE 

result <- bind_rows(result,
                    tibble(Method="Mean + bi",
                           RMSE=RMSE(test_set$rating, pred_bi)))

# intermediate RMSE Improvement result check  

print.data.frame(result)

# including user effect for rating distribution generation 

bu <- train_set %>%
  left_join(bi, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating-mu-b_i))

# prediction including user effect 

pred_busr <- test_set %>%
  left_join(bi, by="movieId") %>%
  left_join(bu, by="userId") %>%
  mutate(pred=mu + b_i + b_u) %>%
  .$pred

# updating the result in the table 

result <- bind_rows(result,
                    tibble(Method="Mean + bi + bu",
                           RMSE=RMSE(test_set$rating,pred_busr)))

# intermediate results RMSE Improvement 

print.data.frame(result)

# plot user effect distribution 

train_set %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  filter(n() >= 100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(color="black") +
  ggtitle("User Effect Distribution") +
  xlab("User Bias") +
  ylab("Count") +
  scale_y_continuous(labels=comma) +
  theme_grey()

# evaluating the model result 

# identigy the 10 largest residual differences 

train_set %>%
  left_join(bi,by="movieId") %>%
  mutate(residual=rating-(mu+b_i)) %>%
  arrange(desc(abs(residual))) %>%
  slice(1:10)

titles <- train_set %>%
  select(movieId, title) %>%
  distinct()

# identifying top 10 best movies 

bi %>%
  inner_join(titles, by="movieId") %>%
  arrange(-b_i) %>%
  select(title) %>%
  head()

# identifying top 10 worst movies 

bi %>%
  inner_join(titles,by="movieId") %>%
  arrange(b_i) %>%
  select(title) %>%
  head()

# identifying number of ratings for 10 best movies 

train_set %>% 
  left_join(bi, by="movieId") %>%
  arrange(desc(b_i)) %>%
  group_by(title) %>%
  summarise(n=n()) %>%
  slice(1:10)

train_set %>% count(movieId) %>%
  left_join(bi,by="movieId") %>%
  arrange(desc(b_i)) %>%
  slice(1:10) %>%
  pull(n)
  
# Regularization the user and movies effects calculation 

regularization <- function(lambda, trainset, testset){
  # Mean calculation 
  mu <- mean(trainset$rating)
  
  # predicting Movie Effect (bi)
  b_i <- trainset %>%
    group_by(movieId) %>%
    summarise(b_i=sum(rating-mu)/(n()+lambda))
  
  # predicting user effect calculation (bu)
  b_u <- trainset %>%
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u=sum(rating-b_i-mu)/(n()+lambda))
  
  # prediction calculation mu + bi + bu
  prediction_ratings <- testset %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(prediction_ratings, testset$rating))
  
}

# defining the set of lambdas to tune 

lambdas <- seq(0,10,0.25)


# tuning the lambda 

rmses <- sapply(lambdas,
                regularization,
                trainset=train_set,
                testset=test_set)

# plot the lambda vs RMSE 

tibble(Lambda=lambdas, RMSE=rmses) %>%
  ggplot(aes(x=Lambda, y=RMSE)) +
  geom_point() +
  ggtitle("Regularization", subtitle = "Pick the penalization that gives the lowest RMSE.") +
  theme_grey()

# picking up the Lambda which will returns Lowest RMSE

lambda <- lambdas[which.min(rmses)]

# predicting rating using parameters achieved through regularization model 

mu <- mean(train_set$rating)

# predicting Movie Effect - bi 

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# predicting user effect - bu

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# prediction regularization model output 

reg_model <- test_set %>%
  left_join(b_i,by="movieId") %>%
  left_join(b_u,by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# predicting regularization bi and bu 

result <- bind_rows(result, 
                    tibble(Method = "Regularized bi and bu", 
                           RMSE = RMSE(test_set$rating, reg_model)))

# intermediate results showing  improvement for RMSE

print.data.frame(result)


#### final validation #####
# linear model with regularization output

mu_edx <- mean(edx$rating)

# predicting Movie effect (bi)
b_i_edx <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# predicting User effect (bu)
b_u_edx <- edx %>% 
  left_join(b_i_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

# final edx vs validation prediction
final_edx <- validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>% 
  pull(pred)

# Updating the results table

result <- bind_rows(result, 
                    tibble(Method = "Final Regularization (edx vs validation)", 
                           RMSE = RMSE(validation$rating, final_edx)))

# Show final result with RMSE achieved or not

print.data.frame(result)