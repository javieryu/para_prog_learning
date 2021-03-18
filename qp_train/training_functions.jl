using LinearAlgebra, JuMP, GLPK, Ipopt, Random, COSMO, StatsBase#, ProfileView

function solve_COSMO(Q, q, A, b)
	model = Model(COSMO.Optimizer)
	set_optimizer_attribute(model, "max_iter", 10000)
	set_optimizer_attribute(model, "eps_abs", 1e-2)
	set_optimizer_attribute(model, "verbose", true)
	@variable(model, y[1:length(q)])
	@objective(model, Min, 0.5 * y' * Q * y + dot(q, y))
	@constraint(model, con_μ[i in 1:length(b)],  dot(A[i,:], y) <= b[i])
	optimize!(model)
	if termination_status(model) == MOI.OPTIMAL
		return value.(y), dual.(con_μ)
	else
		@show termination_status(model)
		error("QP error!")
	end
end

function solve_COSMO2(Q, q, A, b, x0)
	settings = COSMO.Settings(eps_abs=1e-2)
	model = COSMO.Model()
	cn = COSMO.Constraint(-A, b, COSMO.Nonnegatives)
	COSMO.assemble!(model, Q, q, cn, x0=x0, settings=settings)
	res = COSMO.optimize!(model)
	
	return res.x, res.y
end


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

function sketch_hessian(Q::Matrix{Float64}, t::Float64, A::Matrix{Float64}, r::Vector{Float64}, x::Vector{Float64}, sketch_dim::Int64, ps::Vector{Float64})
	Ones = sample(1:length(r), Weights(ps ./ abs.(r)), sketch_dim)
	# Ones = [rand(1:length(r)) for i in 1:sketch_dim]
	H_sketch = Diagonal(1 ./ abs.(r[Ones]))*A[Ones, :]
	return t*Q + H_sketch'*H_sketch
end

# Solve Chebyshev Center LP
function cheby_lp(A, b, ps, sketch_type)
	model = Model(GLPK.Optimizer)
	#set_optimizer_attribute(model, "tol_obj", 1e-2)
	#set_optimizer_attribute(model, "tm_lim", 10000)
	@variable(model, 0 ≤ r ≤ 1)
	@variable(model, x_c[1:size(A, 2)])
	@objective(model, Max, r)

	if sketch_type == :Row_Norm
		for i in 1:length(b)
			@constraint(model, dot(A[i,:],x_c) + r*ps[i] ≤ b[i])
		end
	else
		for i in 1:length(b)
			@constraint(model, dot(A[i,:],x_c) + r*norm(A[i,:]) ≤ b[i])
		end
	end

	optimize!(model)

	if termination_status(model) == MOI.OPTIMAL
		return value.(x_c), value.(r)
	elseif termination_status(model) == MOI.TIME_LIMIT
		isfeas = is_feasible(A, b, value.(x_c))
		println("Sol feasiblility ", isfeas)
		
		if isfeas
			return value.(x_c), value.(r)
		else
			error("CHEB: Couldn't find a feasible point in time.")
		end
	else
		@show termination_status(model)
		error("Chebyshev center error!")
	end
end

function feasibility_lp(A, b)
	model = Model(GLPK.Optimizer)
	@variable(model, x[1:size(A, 2)])
	@objective(model, Min, 0)
	@constraint(model, A*x .≤ b)
	
	optimize!(model)
	
	return value.(x), 1.0
end

# 3 < μ < 100 act roughly the same according to cvx book
function newton_sketch(Q, q, A, b, sketch_dim, sketch_type, x0, rc; β=2.0, tol=1e-3, μ=10)
	if sketch_type == :Uniform
		ps = ones(length(b))
	elseif sketch_type == :Row_Norm
		ps = [norm(A[i,:]) for i in 1:length(b)]
	elseif sketch_type == :Full
		ps = similar(b)
	else
		error("Invalid Sketch Type")
	end

	#@time x, rc = cheby_lp(A, b, ps, sketch_type)
	#@time x, rc = feasibility_lp(A, b)
	x = x0
	m = length(b)
	t = 1.
	
	x_old = x # Clean up after debugging
	while true 
		# println("----------------- Inner Iterations --------------------")
		x = newton_subproblem(x, Q, q, A, b, t, rc, β, ps, sketch_type, sketch_dim)
		if m / t < tol # if t=m/tol then y_opt is within tol of the true solution 
			break
		end
		# println("x norm diff: ", norm(x_old - x))# Clean up after debugging
		x_old = x# Clean up after debugging
		t *= μ
	end
	
	# println("--------------------- Done Ours ------------------------------")
	λ = [-1 / (t*(A[i,:]⋅x - b[i])) for i in 1:length(b)]
	return x, λ
end

function is_feasible(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64})
	for i = 1:size(A, 1)
		if dot(A[i, :], x) - b[i] ≥ 0.0
			return false
		end
	end
	return true
end

# minimize t*x'Qx + t*q'x + log_barrier
function newton_subproblem(x::Vector{Float64}, Q::Matrix{Float64}, q::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, t::Float64, rc::Float64, β::Float64, ps::Vector{Float64}, sketch_type, sketch_dim; ϵ=1e-6)
	terminate = false
	i = 1
	x_old, y, r = similar(x), 0.0, similar(b)
	while true
		r = b - A*x

		# compute direction
		∇ = t*Q*x + t*q + 1 ./ A'*r
		if sketch_type == :Full
			H = t*Q + A'*Diagonal([1/(r[i])^2 for i in 1:length(b)])*A
		else
			H = sketch_hessian(Q, t, A, r, x, sketch_dim, ps)
		end
		
		dir = -normalize(vec(H \ ∇))

		# line search
		α = 2.0 * rc
		while true
			x_temp = x + α * dir
			if is_feasible(A, b, x_temp)
				break 
			end
			
			α /= β
		end

		x_old = x
		y_old = eval(Q, q, r, t, x_old)
		while true
			x_temp = x + α*dir
			y = eval(Q, q, vec(b - A*x_temp), t, x_temp)
			if y ≤ y_old
				x = x_temp
				break
			end
			α /= β
		end

		# This condition also holds when the search dir is wrong.
		terminate = Terminate(x_old, y_old, x, y, ∇, i; ϵₐ=1e-7, ϵᵣ=1e-7, ϵ_g=1e-6, max_iters=80)

		if terminate
			# println("Last α: ", α)
			break
		end
		i += 1
	end
	# println("Exited after ", i, " iterations")
	return x
end

eval(Q, q, r, t, x) = t*0.5*x⋅(Q*x) + t*q⋅x - sum(log.(r))


# input current and past iterate & cost and return whether to terminate or continue.
# Absolute improvement, relative improvement, gradient magnitude, max iters
function Terminate(x_old, y_old, x_new, y_new, ∇, i; ϵₐ=1e-4, ϵᵣ=1e-4, ϵ_g=1e-4, max_iters=100)
	if abs(y_old - y_new) < ϵₐ
		# @show y_old
		# @show y_new
		# println("Abs Error: ", abs(y_old - y_new))
		return true
	elseif abs((y_old - y_new)/ y_old) < ϵᵣ
		# @show y_old
		# @show y_new
		# println("Rel Error")
		return true
#	elseif norm(∇) < ϵ_g
#		println("Grad Mag")
#		return true
	elseif i ≥ max_iters
		# println("Max iters")
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


# Compute gradient of loss w.r.t. learnable parameters for one sample
function ∇loss2(ŷ, λ_, y, λ, Q, A, b)
	∂K_ = [Q                A';
		  Diagonal(λ_)*A    Diagonal(A*ŷ - b)]
	∂K = [Q                A';
		  Diagonal(λ)*A    Diagonal(A*y - b)]  

    ∂b̂ = vcat(zeros(length(ŷ), length(λ_)), -Diagonal(λ_))
	∂b = vcat(zeros(length(y), length(λ)), -Diagonal(λ))

	return norm(inv(∂K_)*∂b̂ - inv(∂K)*∂b) / norm(inv(∂K)*∂b)
end

# Cosine Similarity
function ∇loss3(ŷ, λ_, y, λ, Q, A, b)
	∂K_ = [Q                A';
		  Diagonal(λ_)*A    Diagonal(A*ŷ - b)]
	∂K = [Q                A';
		  Diagonal(λ)*A    Diagonal(A*y - b)]  

    ∂b̂ = vcat(zeros(length(ŷ), length(λ_)), -Diagonal(λ_))
	∂b = vcat(zeros(length(y), length(λ)), -Diagonal(λ))
	
	g1 = inv(∂K_) * ∂b̂
	g2 = inv(∂K) * ∂b

	return dot(g1, g2) / (norm(g1) * norm(g2))
end

#n = 200
#nc = 4000
#sketch_dim = 400
#sketch_type = :Row_Norm # :Uniform, :Row_Norm, :Full
#A = randn(nc, n)
#z = randn(n, 1)
#b = vec(A * z + rand(nc))
#
#R = randn(n, n)
#Q = R' * R + 0.01 * I
##q = randn(n)
#q = zeros(n)
#
#@time x0, rc = cheby_lp(A, b, ones(length(b)), :Uniform)
#
#@time x_ours_not, λ_ours_not = newton_sketch(Q, q, A, b, sketch_dim, :Full, x0, rc, tol=1e-4, μ=10)
#@time x_ours, λ_ours = newton_sketch(Q, q, A, b, sketch_dim, sketch_type, x0, rc, tol=1e-4, μ=10)
## @time x_ipopt, λ_ipopt = solve_IPOPT(Q, q, A, b)
#@time x_cosmo, λ_cosmo = solve_COSMO2(Q, q, A, b, x0)
#
#y_ours = 0.5*x_ours⋅(Q*x_ours) + q⋅x_ours
#y_ours_not = 0.5*x_ours_not⋅(Q*x_ours_not) + q⋅x_ours_not
## y_ipopt = 0.5*x_ipopt⋅(Q*x_ipopt) + q⋅x_ipopt
#y_cosmo = 0.5*x_cosmo⋅(Q*x_cosmo) + q⋅x_cosmo
#
#println("Our Cost: ", y_ours)
#println("Our Unsketched Cost: ", y_ours_not)
## println("Ipopt Cost: ", y_ipopt)
#println("Cosmo Cost: ", y_cosmo)
#
##@profview newton_sketch(Q, q, A, b, sketch_dim, tol=1e-3, μ=10)