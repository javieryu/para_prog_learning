using LinearAlgebra, JuMP, GLPK, Ipopt, Random, COSMO, ProfileView

function test_OSQP()
	n = 100
	nc = 10000
	A = randn(nc, n)
	z = randn(n, 1)
	b = A * z + rand(nc, 1)
	
	R = randn(n, n)
	Q = R' * R + 0.01 * I
	q = 1000.0 * randn(n, 1)
	
	#(y, u) = solve_OSQP(Q, q, A, b)
	#(y2, u2) = solve_IPOPT(Q, q, A, b)
	(y3, u3) = solve_COSMO(Q, q, A, b)
	
	@show(maximum(A * y3 - b))
	#@show(maximum(A * y2 - b))
	#@show(maximum(A * y3 - b))
	
	return
end

function solve_COSMO(Q, q, A, b)
	model = Model(COSMO.Optimizer)
	set_optimizer_attribute(model, "max_iter", 10000)
	@variable(model, y[1:length(q)])
	@objective(model, Min, 0.5 * y' * Q * y + dot(q, y))
	@constraint(model, con_μ[i in 1:length(b)],  dot(A[i,:], y) <= b[i])
	optimize!(model)

	if termination_status(model) == MOI.OPTIMAL # I don't like how JuMP doesn't recognize this as being a global solution
		return value.(y), dual.(con_μ)
	else
		@show termination_status(model)
		error("QP error!")
	end
end

# function solve_OSQP(Q, q, A, b)
# 	model = Model(OSQP.Optimizer)
# 	set_optimizer_attribute(model, "verbose", 0)
# 	set_optimizer_attribute(model, "eps_prim_inf", 1e-6)
# 	set_optimizer_attribute(model, "adaptive_rho", 0)
# 	@variable(model, y[1:length(q)])
# 	@objective(model, Min, 0.5 * y' * Q * y + dot(q, y))
# 	@constraint(model, con_μ[i in 1:length(b)],  dot(A[i,:], y) <= b[i])
# 	optimize!(model)

# 	if termination_status(model) == MOI.OPTIMAL # I don't like how JuMP doesn't recognize this as being a global solution
# 		return value.(y), dual.(con_μ)
# 	else
# 		@show termination_status(model)
# 		error("QP error!")
# 	end
# end

function solve_IPOPT(Q, q, A, b)
	model = Model(Ipopt.Optimizer)
	set_optimizer_attribute(model, "print_level", 0)
	@variable(model, x[1:length(q)])
	@objective(model, Min, 0.5 * x' * Q * x + dot(q, x))
	@constraint(model, con_λ[i in 1:length(b)],  dot(A[i,:], x) <= b[i])
	optimize!(model)

	if termination_status(model) == MOI.LOCALLY_SOLVED
		return value.(x), dual.(con_λ)
	else
		@show termination_status(model)
		error("QP error!")
	end
end



function sketch_hessian(Q::Matrix{Float64}, t::Float64, A::Matrix{Float64}, r::Vector{Float64}, x::Vector{Float64}, sketch_dim::Int64)
	H_sqrt = Diagonal(r)*A
	S = S_count(sketch_dim, length(r))
	H_sketch = S*H_sqrt
	return t*Q + H_sketch'*H_sketch
end

function S_count(sketch_dim, n)
	return S = [rand(1:n) == j for i in 1:sketch_dim, j in 1:n] # one nonzero per row to sample rows from root_hess
end


# Solve Chebyshev Center LP
function cheby_lp(A, b; presolve=false)
	dim = size(A,2)
	model = Model(GLPK.Optimizer)
	presolve ? set_optimizer_attribute(model, "presolve", 1) : nothing
	@variable(model, 0 ≤ r ≤ 1e2)
	@variable(model, x_c[1:dim])
	@objective(model, Max, r)

	for i in 1:length(b)
		@constraint(model, dot(A[i,:],x_c) + r*norm(A[i,:]) ≤ b[i])
	end

	optimize!(model)

	if termination_status(model) == MOI.OPTIMAL
		return value.(x_c)
	elseif termination_status(model) == MOI.NUMERICAL_ERROR && !presolve
		return cheby_lp(A, b, presolve=true)
	else
		@show termination_status(model)
		error("Chebyshev center error!")
	end
end


# 3<μ100 act roughly the same according to cvx book
function newton_sketch(Q, q, A, b, sketch_dim; tol=1e-3, μ=10)
	x = cheby_lp(A, b)
	m = length(b)
	t = 1. 
	
	while true 
		x = newton_subproblem(x, Q, q, A, b, t)
		if m / t < tol # if t=m/tol then y_opt is within tol of the true solution 
			break
		end
		t *= μ
	end
	λ = [-1 / t*(A[i,:]⋅x - b[i]) for i in 1:length(b)]
	return x, λ
end

# minimize t*x'Qx + t*q'x + log_barrier
function newton_subproblem(x::Vector{Float64}, Q::Matrix{Float64}, q::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, t::Float64; ϵ=1e-6)
	terminate = false
	i = 1
	x_old, y = similar(x), 0.0
	while !terminate
		r = b - A*x

		# compute direction
		∇ = t*Q*x + t*q + 1 ./ A'*r
		H = sketch_hessian(Q, t, A, r, x, sketch_dim)
		# H = Q + A'*Diagonal([1/(b[i] - dot(A[i,:], x))^2 for i in 1:length(b)])*A
		dir = -normalize(vec(H \ ∇))

		# line search
		α = α_upper(A, b, x, dir) - ϵ
		x_old = x
		y_old = eval(Q, q, r, t, x_old)
		while true
			x_temp = x + α*dir
			y = eval(Q, q, b - A*x_temp, t, x_old)
			if y ≤ y_old
				x = x_temp
				break
			end
			α /= 2.
		end
		terminate = Terminate(x_old, y_old, x, y, ∇, i; ϵₐ=1e-4, ϵᵣ=1e-4, ϵ_g=1e-4, max_iters=20)
		i += 1
	end
	return x
end

eval(Q, q, r, t, x) = t*0.5*x⋅(Q*x) + t*q⋅x - sum(log.(r))


# input current and past iterate & cost and return whether to terminate or continue.
# Absolute improvement, relative improvement, gradient magnitude, max iters
function Terminate(x_old, y_old, x_new, y_new, ∇, i; ϵₐ=1e-4, ϵᵣ=1e-4, ϵ_g=1e-4, max_iters=100)
	if abs(y_old - y_new) < ϵₐ
		println("Abs Error")
		return true
	elseif abs((y_old - y_new)/ y_old) < ϵᵣ
		println("Rel Error")
		return true
	elseif norm(∇) < ϵ_g
		println("Grad Mag")
		return true
	elseif i ≥ max_iters
		println("Max iters")
		return true
	end
	return false
end

# Find maximum feasible step size
function α_upper(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}, dir::Vector{Float64}; presolve=false)
	model = Model(GLPK.Optimizer)
	@variable(model, 0. ≤ α ≤ 1e3)
	@objective(model, Max, α)
	@constraint(model, α*A*dir .≤ b - A*x)
	optimize!(model)

	if termination_status(model) == MOI.OPTIMAL
		return value(α)
	else
		@show termination_status(model)
		error("Line Search Error")
	end
end




# Compute gradient of loss w.r.t. learnable parameters for one sample
function ∇loss(x, ŷ, y_true, λ, λ˜, Q, A, b, S, G, h, T)
	dₒ, n, n˜ = length(ŷ), length(λ), length(λ˜)

	∂K = [Q                A'                        G';
		  Diagonal(λ)*A    Diagonal(A*ŷ - b - S*x)   zeros(n,n˜);
		  Diagonal(λ˜)*G   zeros(n˜,n)               Diagonal(G*ŷ - h - T*x)]

  	∂l = vcat(2*(ŷ-y_true), zeros(length(λ)+length(λ˜)))

	D =  vec(-∂l' * inv(∂K))
	dz = D[1:dₒ]
	dλ = D[dₒ+1:dₒ+n]

	∇_Q =  0.5*(dz*ŷ' + ŷ*dz')
	∇_q =  dz
	∇_A =  Diagonal(λ)*(dλ*ŷ' + λ*dz')
	∇_b = -Diagonal(λ)*dλ
	∇_S = -Diagonal(λ)*dλ*x'
	return ∇_Q, ∇_q, ∇_A, ∇_b, ∇_S
end


n = 10
nc = 1000
sketch_dim = 100
# A = randn(nc, n)
# z = randn(n, 1)
# b = A * z + rand(nc, 1)

# R = randn(n, n)
# Q = R' * R + 0.01 * I
# q = 1000.0 * randn(n, 1)


# @time x_ipopt, λ_ipopt = solve_IPOPT(Q, q, A, b)
# @time x_ours, λ_ours = newton_sketch(Q, q, A, b, sketch_dim, tol=1e-3, μ=10)

# y_ipopt = 0.5*x_ipopt⋅(Q*x_ipopt) + q⋅x_ipopt
# y_ours = 0.5*x_ours⋅(Q*x_ours) + q⋅x_ours

# println("Ipopt Cost: ", y_ipopt)
# println("Our Cost: ", y_ours)


@profview newton_sketch(Q, q, A, b, sketch_dim, tol=1e-3, μ=10)