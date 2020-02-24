###1 install Packages
rm(list = ls())
#install.packages('readr')
#install.packages("tm")  # for text mining
#install.packages("SnowballC") # for text stemming
#install.packages("wordcloud") # word-cloud generator 
#install.packages("RColorBrewer") # color palettes
# Load packages
library(readr)
require("NLP")
#install.packages("openNLP")
require("openNLP")
library("tm")
library("SnowballC")
library("RColorBrewer")
library("wordcloud")
# read reviews

s1 <- read_csv("~/Desktop/all/2019 FALL B/CIS 434/hw2/adjective/11.txt", col_names = FALSE)
s1 <- paste(s1,collapse = "")
s1 <- as.String(s1)
## Sentence 
sent_token_annotator <- Maxent_Sent_Token_Annotator()
a1 <- annotate(s1, sent_token_annotator) # the first is sentence test and the second function to analyze the anatation 
#a1
## word
word_token_annotator <- Maxent_Word_Token_Annotator()
a2 <- annotate(s1, word_token_annotator, a1)
#a2
annotate(s1, Maxent_Word_Token_Annotator(probs = TRUE), a1) #showing probablity 

### POS Tag
pos_tag_annotator <- Maxent_POS_Tag_Annotator()
a3 <- annotate(s1, pos_tag_annotator, a2) # you need to complete sentence and word segmentation first!!!!!!
#a3
#get adj
a3w <- subset(a3, type == "word") #WE SUBSET THE 
#a3w
tags <- sapply(a3w$features, "[[", "POS") #a list of all those things [[ it is an index function: destract value of a component get the actual value 
#tags
#table(tags)
mypos = a3w[tags=="JJ"]
mypos
adj<-s1[mypos]
adj
#typeof(mypos)



# Load the data as a corpus
docs <- Corpus(VectorSource(adj))
inspect(docs)
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 20)

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=20, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


###2 co-occurrence analysis
library(text2vec)

text1 <- readLines("~/Desktop/all/2019 FALL B/CIS 434/hw2/ratemds1line.txt")
##(1)
it = itoken(text1 , preprocessor=tolower, tokenizer=word_tokenizer) 
# note the default tokenizer is space_tokenizer
vocab <- create_vocabulary(it)
vocab
vocab2 <- prune_vocabulary(vocab, term_count_min = 10L)
vocab2

###(2)
# vocab_vectorizer() creates an object defining how to transform list of tokens into vector space - i.e. how to map words to indices
vectorizer <- vocab_vectorizer(vocab2) 

# Set context window size to 5. The suffix L indicates integer.
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
tcm

####3
target = "great"
target2 = "nice"
target3 = "horrible"
cosim=function(x,y) sum(x*y)/(norm(matrix(x,1),'f') * norm(matrix(y,1),'f'))

m = as.matrix(tcm) # step 1 > 
m = m + t(m) - diag(diag(m)) # step 2 > 
marginal = matrix(rowSums(m), dim(m)[1]) # step 3


sort(tcm[target,], decreasing=TRUE)[1:5]
cosim(tcm[target,],tcm[target2,]) #0.7035128
cosim(tcm[target,],tcm[target3,]) #0.8916867


####4
lift = sum(m) * m / ( marginal %*% t(marginal) )
ppmi = log2( lift*(lift>1) + (lift<=1) )
ppmi

sort(ppmi[target,], decreasing=TRUE)[1:5]

cosim(ppmi[target,],ppmi[target2,]) # 0.2511574
cosim(ppmi[target,],ppmi[target3,]) # 0.1515073
