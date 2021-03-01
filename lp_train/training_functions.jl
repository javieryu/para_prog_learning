using LinearAlgebra, JuMP, GLPK, Random

# TODO: Make LP params a struct instead of passing c, A, b, S, G, h, T to everything

# Compute optimal solution to an LP. Returns y_opt, μ_opt, μ˜_opt
function solve_LP(x, c, A, b, S, G, h, T; presolve=false)
	model = Model(GLPK.Optimizer)
	presolve ? set_optimizer_attribute(model, "presolve", 1) : nothing
	@variable(model, y[1:length(c)])
	@objective(model, Min, c⋅y)
	@constraint(model, con_μ[i in 1:length(b)],  A[i,:]⋅y <= b[i] + S[i,:]⋅x)
	@constraint(model, con_μ˜[i in 1:length(h)], G[i,:]⋅y <= h[i] + T[i,:]⋅x)
	optimize!(model)

	if termination_status(model) == MOI.OPTIMAL
		return value.(y), dual.(con_μ), dual.(con_μ˜)
	elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
		return solve_LP(x, c, A, b, G, h, S, T; presolve=true)
	else
		@show termination_status(model)
		error("LP error!")
	end
end

# computes loss over all data points. Not dividing by num_samples
function mse_loss(X, Y, c, A, b, S, G, h, T)
	loss = 0
	for i in 1:length(X)
		ŷ, nothing, nothing = solve_LP(X[i], c, A, b, S, G, h, T)
		loss += norm(ŷ - Y[i])^2
	end
	return loss
end


# Compute gradient of loss w.r.t. learnable parameters for one sample
function ∇loss(x, ŷ, y_true, μ, μ˜, A, b, S, G, h, T)
	dₒ, n, n˜ = length(ŷ), length(μ), length(μ˜)

	∂K = [zeros(dₒ,dₒ)     A'                        G';
		  Diagonal(μ)*A    Diagonal(A*ŷ - b - S*x)   zeros(n,n˜);
		  Diagonal(μ˜)*G   zeros(n˜,n)               Diagonal(G*ŷ - h - T*x)]
  	∂l = vcat(2*(ŷ-y_true), zeros(length(μ)+length(μ˜)))

	D =  ∂l * inv(∂K)
	dz = D[1:dₒ]
	dμ = D[dₒ+1:dₒ+n]

	∇_c = dz
	∇_A = vec(Diagonal(μ)*(dμ*ŷ' + μ*dz')) # use vec() to flatten and reshape(∇_A, size(A)) to restore to matrix
	∇_b = -Diagonal(μ)*dμ
	∇_S = vec(-Diagonal(μ)*dμ*x')
	return vcat(∇_c, ∇_A, ∇_b, ∇_S)
end

# perform a gradient descent update on the learnable parameters. Updates parameters in place
function update_params!(c, A, b, S, α, ∇)
	dir = normalize(∇)
	c_range = 1 : length(c)
	A_range = c_range[end]+1 : c_range[end]+length(A)
	b_range = A_range[end]+1 : A_range[end]+length(b)
	S_range = b_range[end]+1 : length(∇)

	c -= α*dir[c_range]
	A -= α*reshape(dir[A_range], size(A))
	b -= α*dir[b_range]
	S -= α*reshape(dir[S_range], size(S))
	return nothing
end


# optimize parameters. batch_size=1 is vanilla SGD.
function Train(X, Y, batch_size, epochs, α)
	dᵢ, dₒ = length(X[1]), length(Y[1])
	n = 5
	c = randn(dₒ)
	A = randn(n, dₒ)
	b = A*randn(dₒ) + 10*rand(n)
	# S = randn(n, dᵢ)
	S = 0.0001*Matrix{Float64}(I, n, dᵢ)
	G = vcat(Matrix{Float64}(I, dₒ, dₒ), -Matrix{Float64}(I, dₒ, dₒ))
	h = 5*vcat(ones(dₒ), -ones(dₒ))
	T = zeros(length(h),dᵢ)
	num_params = dₒ + n*dₒ + n + n*dᵢ
	num_batches = Int(ceil(length(X) / batch_size))
	batches = [i != num_batches ? ((i-1)*batch_size+1:i*batch_size) : ((i-1)*batch_size+1:length(X)) for i in 1:num_batches]
	loss = zeros(num_batches*epochs)
	
	iter = 1
	for e in 1:epochs
		# need to shuffle data here. Something like: a = a[shuffle(1:end), :]
		for batch in batches
			∇ = zeros(num_params)
			for i in batch
				ŷ, μ, μ˜ = solve_LP(X[i], c, A, b, S, G, h, T)
				∇ += ∇loss(X[i], ŷ, Y[i], μ, μ˜, A, b, S, G, h, T) # should I divide ∇ by length(batch)?
			end
			update_params!(c, A, b, S, α, ∇)
			loss[iter] = mse_loss(X, Y, c, A, b, S, G, h, T)
			iter += 1
		end
	end
	return c, A, b, S, loss
end






