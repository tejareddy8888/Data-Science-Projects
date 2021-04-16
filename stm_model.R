
setwd('C:/Users/potta/Documents/RA/BLM_Sentiment_analysis')

library(stm)
library(igraph)
library(stmCorrViz)
library(lubridate)

data <- read.csv("second_processed_tweets.csv")
#data <- data[data$continent!='',]
#rownames(data) <- seq(length=nrow(data))
 
#data$publish_time <- as.Date(data$publish_time, format="%Y-%m-%d")
#data$day <- yday(data$publish_time)
# load("complete.RData")

processed <- textProcessor(data$clean_tweet, metadata=data)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

docs <- out$documents
vocab <- out$vocab
meta <- out$meta

# plotRemoved(processed$documents, lower.thresh=seq(1,200, by=100))

# selectedModel2 <- stm(out$documents, out$vocab, K=7, prevalence=~s(day),
#                        max.em.its=20, data=out$meta, init.type="Spectral",
#                        seed=8458159, emtol=0.000015)
# 
# plot(poliblogPrevFit, type="summary", xlim=c(0,.4))
# plot(poliblogPrevFit, type="labels", topics=c(3,7,20))
# plot(poliblogPrevFit, type="hist")
# plot(poliblogPrevFit, type="perspectives", topics=c(7,10))
# topicQuality(model=poliblogPrevFit, documents=docs)
# plot(poliblogPrevFit$convergence$bound, type="l", ylab="Approximate Objective", 
#      main="Convergence")

#kResult <- searchK(out$documents, out$vocab, K=c(10,15,20,25), prevalence=~day, data=meta)



poliblogSelect <- selectModel(out$documents, out$vocab, K=10, prevalence=~day,
                              max.em.its=133, data=meta, runs=20, seed=8458159)
plotModels(poliblogSelect)
selectedModel2 <- poliblogSelect$runout[[1]]

plot(selectedModel2, type="summary", topics=c(5,4,1,7), xlim=c(0,.4))
plot(selectedModel2, type="labels", topics=c(5,4,1,7))
plot(selectedModel2, type="hist",topics=c(5,4,1,7))
plot(selectedModel2, type="perspectives",topics=c(5,4))
plot(selectedModel2, type="perspectives",topics=c(1,7))
topicQuality(model=selectedModel2, documents=docs)



# storage <- manyTopics(out$documents, out$vocab, K=c(3,5,7,10,15), prevalence=~week,
#                       data=meta, runs=10)
# storageOutput1 <- storage$out[[1]] # For exchoosing the model with 7 topics
# plot(storageOutput1)


## Final Model results
labelTopicsSel <- labelTopics(selectedModel2, c(1,3,5))
print(sageLabels(selectedModel2))

thoughts2 <- findThoughts(selectedModel2, texts=as.character(meta$clean_tweet), n=5, topics=8)$docs[[1]]
print(thoughts2)
thoughts6 <- findThoughts(selectedModel2, texts=as.character(meta$clean_tweet), n=2, topics=3)$docs[[1]]
print(thoughts6)
thoughts7 <- findThoughts(selectedModel2, texts=as.character(meta$clean_tweet), n=9, topics=10)$docs[[1]]
print(thoughts7)


# plotQuote(thoughts3, width=40, main="Topic 3")



# out$meta$rating <- as.factor(out$meta$rating)
prep <- estimateEffect(1:10 ~ day, selectedModel2, meta=out$meta,  uncertainty="Global")

plot(prep, "day", method="continuous", topics=c(1:10), model=selectedModel2, printlegend=TRUE, xaxt="n", 
     xlab="Time")

plot(prep, "day", method="continuous", topics=c(5,4,1,7), model=selectedModel2, printlegend=TRUE, xaxt="n", 
     xlab="Time")

# plot(prep, covariate="continent", topics=c(1:7), model=selectedModel2, 
#      method="difference", cov.value1="Asia", cov.value2="Europe",
#      xlab="Europe ... Asia", main="Effect of Asia vs. Europe",
#      xlim=c(-.15,.30))


mod.out.corr <- topicCorr(selectedModel2)
plot(mod.out.corr)