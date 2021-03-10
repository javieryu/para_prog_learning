using Plots
include("training_functions.jl")

# Setup training and testing data
bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]
n_samples = 50
X = [[bound_r(-2*π,2*π)] for i in 1:n_samples]
Y = [sin.(X[i]) for i in 1:length(X)] # Trying to learn y=sin(x) for -5 ≤ x ≤ 5

# Set hyperparameters
batch_size = 1
epochs = 1
α = 0.001 # step size

# Set (unchanging) input-output constraints
dᵢ, dₒ = length(X[1]), length(Y[1])
G = vcat(Matrix{Float64}(I, dₒ, dₒ), -Matrix{Float64}(I, dₒ, dₒ))
h = 1*vcat(ones(dₒ), ones(dₒ))
T = zeros(length(h),dᵢ)

U, q, A, p, Δ, S, loss = Train(X, Y, G, h, T, batch_size, epochs, α)

# add plot
Ŷ = [zeros(dₒ) for i in 1:length(X)]
for i in 1:length(X)
	Ŷ[i], nothing, nothing = solve_QP(X[i], U, q, A, p, Δ, S, G, h, T)
end

plt = scatter(vcat(X...), vcat(Ŷ...), label="pQP")
scatter!(plt, vcat(X...), vcat(Y...), label="True")
