weights <- read.delim("/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/linear.out", sep=',', header = FALSE)
N = length(weights[,1])

r = max(abs(weights))

plot(1:N,weights[,1], ylim=c(-r,r))
points(1:N, weights[,2], col='red')
points(1:N, weights[,3], col='blue')

for(i in 1:3){
  lines(c(i*5+0.5, i*5+0.5), c(-r,r))
}
for(i in 1:5){
  lines(c(15 + i*10+0.5, 15 + i*10+0.5), c(-r,r))
}

# for(i in 0:4){
#   plot(1:3, c(weights[1,i*5+1], weights[6,1], weights[11,1]), type='l', ylim=c(-1,1))
#   lines(1:3, c(weights[1,2], weights[6,2], weights[11,2]), col='red')
#   lines(1:3, c(weights[1,3], weights[6,3], weights[11,3]), col='blue')
# }
