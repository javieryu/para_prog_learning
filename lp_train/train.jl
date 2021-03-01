using Plots
include("training_functions.jl")

# Setup training and testing data
bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]
n_samples = 100
X = [[bound_r(-5,5)] for i in 1:n_samples]
Y = [sin.(X[i]) for i in 1:length(X)] # Trying to learn y=sin(x) for -5 ≤ x ≤ 5

# Set hyperparameters
batch_size = 1
epochs = 1
α = 0.1 # step size

c, A, b, S, loss = Train(X, Y, batch_size, epochs, α)


