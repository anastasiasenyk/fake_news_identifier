library(tidytext)
library(readr)
library(ggplot2)
library(dplyr)
library(tokenizers)
library(plotrix)
library(dplyr)




naiveBayes <- setRefClass("naiveBayes",
                          
                          # here it would be wise to have some vars to store intermediate result
                          # frequency dict etc. Though pay attention to bag of wards! 
                          fields = list(
                            credible_probs = "table",
                            fake_probs = "table",
                            credible_news = "data.frame",
                            fake_news = "data.frame",
                            p_credible = "table",
                            p_fake = "table"
                            
                          ),
                          
                          
                          
                          
                          methods = list(
        
                            tokens_conditional_prob = function(tokens) {
                              occurrences <- table(unlist(tokens)) + 1
                              return (occurrences / sum(occurrences))
                            },
                            
                            
                            
                            
                            # prepare your training data as X - bag of words for each of your
                            # messages and corresponding label for the message encoded as 0 or 1 
                            # (binary classification task)
                            fit_data = function(dataf, splitted_stop_words)
                            {
                              dataf$text <- paste(dataf$Headline,dataf$Body, sep = " ")
                              dataf$text <- tokenize_words(dataf$text, lowercase = TRUE, strip_numeric = TRUE, strip_punct = TRUE, stopwords = splitted_stop_words)
                              
                              
                              credible_news <<- subset(dataf, Label == 'credible')
                              fake_news <<- subset(dataf, Label == 'fake')
                              
                              credible_probs <<- tokens_conditional_prob(credible_news$text)
                              fake_probs <<- tokens_conditional_prob(fake_news$text)
                              
                              truth_rate <<- credible_probs %>%
                                as.data.frame() %>% 
                                arrange(desc(Freq))
                              
                              fake_rate <<- fake_probs %>%
                                as.data.frame() %>% 
                                arrange(desc(Freq))
                              
                              truth_top <<- truth_rate[1:5, ]

                              truth_top$Fake <<- c(0, 0, 0, 0, 0)
                              
                              for (j in 1:5){
                                for (i in 1:nrow(fake_rate)){
                                  if (fake_rate[i,1] == truth_top[j,1]){
                                    truth_top[j, 3] <<- fake_rate[i,2]
                                  }
                                }
                              }
                              

                              popular <- as.matrix(data.frame(will = c(truth_top[1, 2], truth_top[1, 3]),
                                                           can = c(truth_top[2, 2], truth_top[2, 3]),
                                                           one = c(truth_rate[3, 2], truth_top[3, 3]),
                                                           game = c(truth_top[4, 2], truth_top[4, 3]),
                                                           news = c(truth_rate[5, 2], truth_top[5, 3])))
                              
                              barplot(popular, main = "Probability of top-words",
                                      col = c("green", "red"),
                                      beside = TRUE)
                              legend("topright",
                                legend = c("Credible", "Fake"),
                                     fill = c("green", "red"))
                              
                              p_credible <<- table(credible_news$Label) / sum(table(dataf$Label))
                              p_fake <<- table(fake_news$Label) / sum(table(dataf$Label))
 
                              return (dataf)
                            },
                           
                            
                            
                            
                            fit_test = function(dataf, splitted_stop_words)
                            {
                              dataf$text <- paste(dataf$Headline, dataf$Body, sep = " ")
                              dataf$text <- tokenize_words(dataf$text, lowercase = TRUE, strip_numeric = TRUE, strip_punct = TRUE, stopwords = splitted_stop_words)
                              
                              return (dataf)
                              
                            },
                            
                            
                            
                            
                            # return prediction for a single message 
                            predict_for_dataf = function(dataf)
                            {
                              
                              
                              dataf$res <- NA
                              index = 1
                              
                              for (news in dataf$text) {
                                dataf[index, 6] = predict(news)
                                index = index + 1
                              }
                              
                              dataf <- dataf[c('Headline', 'Body', 'Label', 'res')]
                              return (dataf)
                              
                            },
                            
                            
                            
                            
                            predict = function(news)
                            {
                              credible_guess = p_credible
                              fake_guess = p_fake
                              res = 1
                              
                              for (word in unlist(news)) {
                                credible_word <- credible_probs[word]
                                fake_word <- fake_probs[word]
                                

                                
                                if (!is.na(credible_probs[word]) & !is.na(fake_probs[word])) {
                                  credible_guess = (credible_probs[word] * p_credible) / (credible_probs[word] * p_credible + fake_probs[word] * p_fake)
                                }

                                if (!is.na(fake_probs[word]) & !is.na(credible_probs[word])) {
                                  fake_guess = (fake_probs[word] * p_fake) / (credible_probs[word] * p_credible + fake_probs[word] * p_fake)
                                }
                                
                                
                                res = res * (credible_guess / fake_guess)
                              }
                              
                              if (res > 1) {
                                return ('credible')
                              }
                              return ('fake')
                              },
                            
                            
                            
                            
                            
                            # score you test set so to get the understanding how well you model
                            # works.
                            # look at f1 score or precision and recall
                            # visualize them 
                            # try how well your model generalizes to real world data! 
                            score = function(dataf)
                            {
                              tp = sum(table(which(dataf$Label == 'credible' & dataf$res == 'credible')))
                              fp = sum(table(which(dataf$Label == 'fake' & dataf$res == 'credible')))
                              fn = sum(table(which(dataf$Label == 'credible' & dataf$res == 'fake')))
                              precision = tp / (tp + fp)

                              recall = tp / (tp + fn)

                              f1score = 2 * (precision * recall) / (recall + precision)
                              
                              return f1score
                            },
                            score_visual = function(dataf)
                            {
                                true_positive = table(dataf$res=="credible" & dataf$Label=="credible")["TRUE"]
                                true_negative = table(dataf$res=="fake" & dataf$Label=="fake")["TRUE"]
                                false_negative = table(dataf$res=="fake" & dataf$Label=="credible")["TRUE"]
                                false_positive = table(dataf$res=="credible" & dataf$Label=="fake")["TRUE"]
                                slices <- c(true_positive, true_negative, false_positive, false_negative)
                                lbls <- c("True Positive", "True Negative", "False Positive", "False Negative")
                                pct <- round(slices/sum(slices)*100)
                                lbls <- paste(lbls, pct)
                                lbls <- paste(lbls,"%",sep="")
                                pie(slices, labels = lbls, col=rainbow(length(lbls)), main="Prediction accuracy")
                            }
                          )
                          )



stop_words_path <- "/home/fazhur/R_Projects/Lab1/stop_words.txt"
test_path <- "/home/fazhur/R_Projects/Lab1/2-fake_news/test.csv"
train_path <- "/home/fazhur/R_Projects/Lab1/2-fake_news/train.csv"



main_f <- function(test_path, train_path, stop_words_path){

  test <- read.csv(test_path, stringsAsFactors = FALSE)
  data <- read.csv(train_path, stringsAsFactors = FALSE)
  stop_words <- read_file(stop_words_path)
  splitted_stop_words <- strsplit(stop_words, split='\n')[[1]]

  
  model = naiveBayes()
  data <- model$fit_data(data, splitted_stop_words)
  test <- model$fit_test(test, splitted_stop_words)

  test <- model$predict_for_dataf(test);
  
  print('score - >')
  print(model$score(test))
  print(model$score_visual(test))

  return (test)
}



test <- main_f(test_path, train_path, stop_words_path)




