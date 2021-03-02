using LinearAlgebra, JuMP, GLPK, Ipopt, Random

# TODO: Make QP params a struct instead of passing Q, q, A, b, S, G, h, T to everything

# Compute optimal solution to a QP. Returns y_opt, μ_opt, μ˜_opt
function solve_QP(x, Q, q, A, b, S, G, h, T)
	model = Model(Ipopt.Optimizer)
	set_optimizer_attribute(model, "print_level", 0)
	@variable(model, y[1:length(q)])
	@objective(model, Min, 0.5*y'*Q*y + q⋅y)
	@constraint(model, con_μ[i in 1:length(b)],  A[i,:]⋅y <= b[i] + S[i,:]⋅x)
	@constraint(model, con_μ˜[i in 1:length(h)], G[i,:]⋅y <= h[i] + T[i,:]⋅x)
	optimize!(model)

	if termination_status(model) == MOI.LOCALLY_SOLVED # I don't like how JuMP doesn't recognize this as being a global solution
		return value.(y), dual.(con_μ), dual.(con_μ˜)
	else
		@show termination_status(model)
		error("QP error!")
	end
end

# computes loss over all data points. Not dividing by num_samples
function mse_loss(X, Y, Q, q, A, b, S, G, h, T)
	loss = 0
	for i in 1:length(X)
		ŷ, nothing, nothing = solve_QP(X[i], Q, q, A, b, S, G, h, T)
		loss += norm(ŷ - Y[i])^2
	end
	return loss
end

# Compute gradient of loss w.r.t. learnable parameters for one sample
function ∇loss(x, ŷ, y_true, μ, μ˜, Q, A, b, S, G, h, T)
	dₒ, n, n˜ = length(ŷ), length(μ), length(μ˜)

	∂K = [Q                A'                        G';
		  Diagonal(μ)*A    Diagonal(A*ŷ - b - S*x)   zeros(n,n˜);
		  Diagonal(μ˜)*G   zeros(n˜,n)               Diagonal(G*ŷ - h - T*x)]

  	∂l = vcat(2*(ŷ-y_true), zeros(length(μ)+length(μ˜)))

	D =  vec(-∂l' * inv(∂K))
	dz = D[1:dₒ]
	dμ = D[dₒ+1:dₒ+n]

	∇_Q =  0.5*(dz*ŷ' + ŷ*dz')
	∇_q =  dz
	∇_A =  Diagonal(μ)*(dμ*ŷ' + μ*dz')
	∇_b = -Diagonal(μ)*dμ
	∇_S = -Diagonal(μ)*dμ*x'
	return ∇_Q, ∇_q, ∇_A, ∇_b, ∇_S
end

# perform a gradient descent update on the learnable parameters. Not sure why not updating parameters in place
function update_params!(Q, q, A, b, S, ∇_Q, ∇_q, ∇_A, ∇_b, ∇_S, α)
	Q -= α*reshape(normalize(vec(∇_Q)), size(∇_Q))
	q -= α*normalize(∇_q)
	A -= α*reshape(normalize(vec(∇_A)), size(∇_A))
	b -= α*normalize(∇_b)
	S -= α*reshape(normalize(vec(∇_S)), size(∇_S))
	return Q, q, A, b, S
end


# optimize parameters. batch_size=1 is vanilla SGD.
function Train(X, Y, G, h, T, batch_size, epochs, α)
	dᵢ, dₒ = length(X[1]), length(Y[1])
	n = 5
	U = rand(dₒ, dₒ)
	Q = U'*U + 0.001*I
	q = randn(dₒ)
	A = randn(n, dₒ)
	b = A*randn(dₒ) + 5*rand(n)
	S = 0.01randn(n, dᵢ)

	num_batches = Int(ceil(length(X) / batch_size))
	batches = [i != num_batches ? ((i-1)*batch_size+1:i*batch_size) : ((i-1)*batch_size+1:length(X)) for i in 1:num_batches]
	loss = zeros(num_batches*epochs)
	
	iter = 1
	for e in 1:epochs
		# need to shuffle data here. Something like: a = a[shuffle(1:end), :]
		for batch in batches
			∇_Q, ∇_q, ∇_A, ∇_b, ∇_S = zeros(size(Q)), zeros(length(q)), zeros(size(A)), zeros(length(b)), zeros(size(S))
			for i in batch
				ŷ, μ, μ˜ = solve_QP(X[i], Q, q, A, b, S, G, h, T)
				∇_Qᵢ, ∇_qᵢ, ∇_Aᵢ, ∇_bᵢ, ∇_Sᵢ = ∇loss(X[i], ŷ, Y[i], μ, μ˜, Q, A, b, S, G, h, T) # should I divide ∇ by length(batch)?
				∇_Q += ∇_Qᵢ # find shorter way to do this
				∇_q += ∇_qᵢ
				∇_A += ∇_Aᵢ
				∇_b += ∇_bᵢ
				∇_S += ∇_Sᵢ
			end
			Q, q, A, b, S = update_params!(Q, q, A, b, S, ∇_Q, ∇_q, ∇_A, ∇_b, ∇_S, α)
			loss[iter] = mse_loss(X, Y, Q, q, A, b, S, G, h, T)
			println("Iter: ", iter, "    Loss: ", loss[iter])
			iter += 1
		end
	end
	return Q, q, A, b, S, loss
end






### UNUSED FUNCTIONS ###
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