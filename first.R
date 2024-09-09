score <- read.delim("/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/first.out", sep=' ', header = FALSE)

N = length(score[,1])
L = length(score[1,])
score = score[,-L]
L = L-1

# matplot(score[,1:(L/2)], type="l", col=1:(L/2), pch=0) #plot
# legend("topleft", legend = 1:(L/2), col=1:(L/2), pch=1) # optional legend

# matplot(score[,(L/2+1):L], type="l", col=1:(L/2), pch=0) #plot
# legend("topleft", legend = 1:(L/2), col=1:(L/2), pch=1) # optional legend


print(colSums(score))

plot(score[,1] + score[,3] + score[,7] + score[,9], type='l')
lines(score[,2] + score[,4] + score[,6] + score[,8], col='red')
lines(score[,5], col='blue')