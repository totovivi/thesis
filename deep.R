library(h2oEnsemble) ; library(png)

#TRANSFORM BOARD TO IMAGE
#b <- c(1,2,0,0,2,1,2,2,1)
#b <- c(0,2,1,0,1,1,2,2,1)
b <- c(1,2,2,0,1,0,2,0,1)

setwd('/Users/Thomas/Dropbox/DS/thesis')
X <- t(c(readPNG(paste0(sample(c('X', 'X2', 'X3', 'X4', 'X5')), '.png'))))
O <- t(c(readPNG(paste0(sample(c('O', 'O2', 'O3', 'O4', 'O5')), '.png'))))
N <- t(c(readPNG('N.png')))

T <- function(num) {
	if (num == 1) img <- X
	else if (num == 2) img <- O
	else img <- N
	matrix(img, nrow=50, ncol=50, byrow=F)
}

row1 <- cbind(T(b[1]), T(b[2]), T(b[3]))
row2 <- cbind(T(b[4]), T(b[5]), T(b[6]))
row3 <- cbind(T(b[7]), T(b[8]), T(b[9]))
img <- rbind(row1, row2, row3)

#transformations
transfo <- sample(c(.8, -.1, .3))
img <- img + row(img)*.001*sample(c(rep(-1, 5), 1))*rnorm(22500, -.05, .1)+.05
#img <- img + sample(c(sample(c(1,4,7,15)),1))*rnorm(22500, -.05, .1)

writePNG(img, 'mail1.png')

#NEURAL NETWORK
h2o.init(nthreads = 4, max_mem_size = '12g')

trainWin <- t(c(readPNG('test.png')))
testWin <- t(c(readPNG('test2.png')))
trainLoose <- t(c(readPNG('test3.png')))
testLoose <- t(c(readPNG('test4.png')))
testLoose2 <- t(c(readPNG('test5.png')))

#trainWin2 <- t(c(readPNG('trainWin2.png')))
#trainLost <- t(c(readPNG('trainLost.png')))
#testWin <- t(c(readPNG('testWin.png')))
#testWin2 <- t(c(readPNG('testWin2.png')))
#testLost <- t(c(readPNG('testLost.png')))

train <- data.frame(rbind(trainWin, trainLoose), y=c(1,-1))
write.csv(train, 'trainH2O.csv', row.names=F)
train <- h2o.importFile('/Users/Thomas/Dropbox/DS/thesis/trainH2O.csv')

test <- data.frame(rbind(testWin, testLoose, testLoose2))
write.csv(test, 'testH2O.csv', row.names=F)
test <- h2o.importFile('/Users/Thomas/Dropbox/DS/thesis/testH2O.csv')

y <<- 'y' ; X <<- setdiff(names(train), y)

fit <- h2o.deeplearning(X, y, training_frame=train, hidden = c(50,50,50), epochs=10)

predict(fit, train)
predict(fit, test)



