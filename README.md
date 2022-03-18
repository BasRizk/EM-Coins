# EM-Coins
Implementation of Expectation Maxmization Algorithm (Framework) on example of coin flipping illustrated in

https://ra-training.s3-us-west-1.amazonaws.com/DoBatzoglou_2008_EMAlgo.pdf


##To Run the API(Using GMM): python main.py ##

The API implementation in Flask is simple. Loading the dataset has been implemented using ThreadPoolExecutor to execute as many requests as possible simultaneously to load the dataset (30 coin draws) quickly enough so that there won’t be much delay for the API request to make the estimation.

The EM algorithm is an approach of estimating the unknown parameters behind an observed distribution of data iteratively, in other words it is used in learning the statistical model (latent variables) behind probabilistic distributions such as the means and standard deviations of an observed distribution. Gaussian Mixture Model (GMM) is one probabilistic model which can be developed, meaning its parameters including its components' means can be estimated, using EM algorithm.

Assuming that the probability of picking one of the two coins in the first place 0.5 (fair), the task at hand is fairly simple, and the EM algorithm to solve it can be quickly developed from scratch by applying bayes rule in the E-Step given dummy initial parameters (thetas) and the binomial distribution in each sample to calculate the probability of each coin given each observed trial, then updating those thetas (corresponding to the coin biases) based on maximum likelihood distribution equation over the contributions estimations calculated of each coin in the previous step. This would go on until convergence.

EM algorithm does not necessarily converge to local optimum, so we might run it several times and average our estimations, or make an educated guess of the parameters, or employ an approach of figuring out more realistic initializations, that can be achieved for example by k-means. Scikit-learn has a package that learns GMM (configured by default to use k-means for initializations). I have employed this in my API and set the number of components to 2 corresponding to two probability distributions of the 2 coins. The only thing left is to format the dataset in a way that the model can fit, that is by turning each sample into one point corresponding to the percentage of the heads in it. Hence when the model is fit by iteratively applying EM, it has learned the latent variables, of which are the mean values of the two coins probability distributions, and those correspond to the coin biases (probability of heads per each coin).

Estimates calculated for the two coins probabilities of turning heads are approximately 30% and 70%. 
