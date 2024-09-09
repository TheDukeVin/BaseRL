
score <- read.delim("/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/score.out", sep=',', header = FALSE)
N = length(score)

NUM_AVG = 1

consec_avg <- function(lst, N){
  avgs <- integer(length(lst) / N)
  for(i in 0:(length(lst) / N - 1)){
    x = 0
    count = 0
    for(j in 1:N){
      num = as.numeric(lst[i*N + j])
      x = x + num
      count = count + 1
    }
    avgs[i+1] = x / count
  }
  avgs
}

avg_score = consec_avg(score, NUM_AVG)
# plot(1:length(avg_score), avg_score, ylim=c(0,100))
plot(1:length(avg_score), avg_score)