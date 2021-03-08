using LinearAlgebra, JuMP, GLPK, Ipopt, Random

# TODO: Make QP params a struct instead of passing Q, q, A, b, S, G, h, T to everything

# Compute optimal solution to a QP. Returns y_opt, μ_opt, μ˜_opt
function solve_QP(x, U, q, A, p, Δ, S, G, h, T)
	Q = U'*U + 0.001*I
	Sx = max.(0, S*x)
	b = A*p + Δ

	model = Model(Ipopt.Optimizer)
	set_optimizer_attribute(model, "print_level", 0)
	@variable(model, y[1:length(q)])
	@objective(model, Min, 0.5*y'*Q*y + q⋅y)
	@constraint(model, con_μ[i in 1:length(b)],  A[i,:]⋅y <= b[i] + Sx[i])
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
function mse_loss(X, Y, U, q, A, p, Δ, S, G, h, T)
	loss = 0
	for i in 1:length(X)
		ŷ, nothing, nothing = solve_QP(X[i], U, q, A, p, Δ, S, G, h, T)
		loss += norm(ŷ - Y[i])^2
	end
	return loss
end

# Compute gradient of loss w.r.t. learnable parameters for one sample
function ∇loss(x, ŷ, y_true, μ, μ˜, U, A, p, Δ, S, G, h, T)
	Q = U'*U + 0.001*I
	Sx = max.(0, S*x)
	b = A*p + Δ
	dₒ, n, n˜ = length(ŷ), length(μ), length(μ˜)

	∂K = [Q                A'                        G';
		  Diagonal(μ)*A    Diagonal(A*ŷ - b - Sx)   zeros(n,n˜);
		  Diagonal(μ˜)*G   zeros(n˜,n)               Diagonal(G*ŷ - h - T*x)]

  	∂l = vcat(2*(ŷ-y_true), zeros(length(μ)+length(μ˜)))

	D =  vec(-∂l' * inv(∂K))
	dz = D[1:dₒ]
	dμ = D[dₒ+1:dₒ+n]

	∇_Q =  0.5*(dz*ŷ' + ŷ*dz')
	∇_q =  dz
	∇_A =  Diagonal(μ)*(dμ*ŷ' + μ*dz')
	∇_b = -Diagonal(μ)*dμ
	∇_S = -Diagonal(μ)*dμ # *x'

	∇_U = ∇_Q*2*U
	∇_S = (∇_S .* [Sx[i] > 0 for i in 1:length(Sx)]) * x' # gradient of ReLU
	∇_p = vec(reshape(∇_b, 1, length(∇_b))*A)

	return ∇_U, ∇_q, ∇_A, ∇_p, ∇_S
end

safe_normalize(v) = v == zeros(length(v)) ? (return zeros(length(v))) : (return normalize(v))


# perform a gradient descent update on the learnable parameters. Not sure why not updating parameters in place
function update_params!(U, q, A, p, S, ∇_U, ∇_q, ∇_A, ∇_p, ∇_S, α)
	U -= α*reshape(safe_normalize(vec(∇_U)), size(∇_U))
	q -= α*safe_normalize(∇_q)
	A -= α*reshape(safe_normalize(vec(∇_A)), size(∇_A))
	p -= α*safe_normalize(∇_p)
	S -= α*reshape(safe_normalize(vec(∇_S)), size(∇_S))
	return U, q, A, p, S
end


# optimize parameters. batch_size=1 is vanilla SGD.
function Train(X, Y, G, h, T, batch_size, epochs, α)
	dᵢ, dₒ = length(X[1]), length(Y[1])
	n = 5
	U = rand(dₒ, dₒ)
	Q = U'*U + 0.001*I
	q = randn(dₒ)
	A = randn(n, dₒ)
	p = randn(dₒ)
	Δ = 5*rand(n)
	b = A*p + Δ
	S = 1000*rand(n, dᵢ)

	num_batches = Int(ceil(length(X) / batch_size))
	batches = [i != num_batches ? ((i-1)*batch_size+1:i*batch_size) : ((i-1)*batch_size+1:length(X)) for i in 1:num_batches]
	loss = zeros(num_batches*epochs)
	
	iter = 1
	for e in 1:epochs
		# need to shuffle data here. Something like: a = a[shuffle(1:end), :]
		for batch in batches
			∇_U, ∇_q, ∇_A, ∇_p, ∇_S = zeros(size(U)), zeros(length(q)), zeros(size(A)), zeros(length(p)), zeros(size(S))
			for i in batch
				ŷ, μ, μ˜ = solve_QP(X[i], U, q, A, p, Δ, S, G, h, T)
				∇_Uᵢ, ∇_qᵢ, ∇_Aᵢ, ∇_pᵢ, ∇_Sᵢ = ∇loss(X[i], ŷ, Y[i], μ, μ˜, U, A, p, Δ, S, G, h, T) # should I divide ∇ by length(batch)?
				∇_U += ∇_Uᵢ # find shorter way to do this
				∇_q += ∇_qᵢ
				∇_A += ∇_Aᵢ
				∇_p += ∇_pᵢ
				∇_S += ∇_Sᵢ
			end
			U, q, A, p, S = update_params!(U, q, A, p, S, ∇_U, ∇_q, ∇_A, ∇_p, ∇_S, α)
			loss[iter] = mse_loss(X, Y, U, q, A, p, Δ, S, G, h, T)
			println("Iter: ", iter, "    Loss: ", loss[iter])
			iter += 1
		end
	end
	return U, q, A, p, Δ, S, loss
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