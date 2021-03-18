using FileIO, JLD2, LinearAlgebra
include("training_functions.jl")

# 200, 4000 is what we decided
function gen_data(n, nc)
    A = randn(nc, n)
    z = randn(n, 1)
    b = vec(A * z + rand(nc))
    
    R = randn(n, n)
    Q = R' * R + 0.01 * I

    return Q, A, b
end


# (200, 4000, 12, [10 ,50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
function gen_results(n, nc, iters, sketch_dims)
	time_dict = Dict{String, Dict{Int64, Vector{Float64}} }() # solve type -> sketch_dim -> vector of times
	cost_dict = Dict{String, Dict{Int64, Vector{Float64}} }() # solve type -> sketch_dim -> vector of costs
	norm_dict = Dict{String, Dict{Int64, Vector{Float64}} }() # solve type -> sketch_dim -> vector of norms
	x_dict = Dict{String, Dict{Int64, Matrix{Float64}} }() # solve type -> sketch_dim -> matrix of xs
	λ_dict = Dict{String, Dict{Int64, Matrix{Float64}} }() # solve type -> sketch_dim -> matrix of λs

	for method in ["Full", "Uniform", "Row_Norm", "Cosmo"]
		time_dict[method] = Dict{Int64, Vector{Float64}}()
		cost_dict[method] = Dict{Int64, Vector{Float64}}()
		norm_dict[method] = Dict{Int64, Vector{Float64}}()
		x_dict[method] = Dict{Int64, Matrix{Float64}}()
		λ_dict[method] = Dict{Int64, Matrix{Float64}}()
	end

	for sketch_dim in sketch_dims
		println("\nSketch Dim: ", sketch_dim)
		for method in ["Full", "Uniform", "Row_Norm", "Cosmo"]
			time_dict[method][sketch_dim] = Vector{Float64}(undef, iters)
			cost_dict[method][sketch_dim] = Vector{Float64}(undef, iters)
			norm_dict[method][sketch_dim] = Vector{Float64}(undef, iters)
			x_dict[method][sketch_dim] = Matrix{Float64}(undef, n, iters)
			λ_dict[method][sketch_dim] = Matrix{Float64}(undef, nc, iters)
		end

		for i in 1:iters
			println("Iteration: ", i)
			Q, A, b = gen_data(n, nc)
			x0, rc = cheby_lp(A, b, ones(length(b)), :Uniform)

			(x_full, λ_full), t_full, nothing, nothing, nothing           = @timed newton_sketch(Q, zeros(length(x0)), A, b, sketch_dim, :Full, x0, rc, tol=1e-4, μ=10)
			(x_uniform, λ_uniform), t_uniform, nothing, nothing, nothing  = @timed newton_sketch(Q, zeros(length(x0)), A, b, sketch_dim, :Uniform, x0, rc, tol=1e-4, μ=10)
			(x_norm, λ_norm), t_norm, nothing, nothing, nothing           = @timed newton_sketch(Q, zeros(length(x0)), A, b, sketch_dim, :Row_Norm, x0, rc, tol=1e-4, μ=10)
			(x_cosmo, λ_cosmo), t_cosmo, nothing, nothing, nothing        = @timed solve_COSMO2(Q, zeros(length(x0)), A, b, x0)
			
			
			time_dict["Full"][sketch_dim][i] = t_full
			time_dict["Uniform"][sketch_dim][i] = t_uniform
			time_dict["Row_Norm"][sketch_dim][i] = t_norm
			time_dict["Cosmo"][sketch_dim][i] = t_cosmo

			cost_dict["Full"][sketch_dim][i] = 0.5*x_full⋅(Q*x_full)
			cost_dict["Uniform"][sketch_dim][i] = 0.5*x_uniform⋅(Q*x_uniform)
			cost_dict["Row_Norm"][sketch_dim][i] = 0.5*x_norm⋅(Q*x_norm)
			cost_dict["Cosmo"][sketch_dim][i] = 0.5*x_cosmo⋅(Q*x_cosmo)

			norm_dict["Full"][sketch_dim][i] = ∇loss3(x_full, λ_full, x_cosmo, λ_cosmo, Q, A, b)
			norm_dict["Uniform"][sketch_dim][i] = ∇loss3(x_uniform, λ_uniform, x_full, λ_full, Q, A, b)
			norm_dict["Row_Norm"][sketch_dim][i] = ∇loss3(x_norm, λ_norm, x_full, λ_full, Q, A, b)			

			x_dict["Full"][sketch_dim][:,i] = x_full
			x_dict["Uniform"][sketch_dim][:,i] = x_uniform
			x_dict["Row_Norm"][sketch_dim][:,i] = x_norm
			x_dict["Cosmo"][sketch_dim][:,i] = x_cosmo

			λ_dict["Full"][sketch_dim][:,i] = λ_full
			λ_dict["Uniform"][sketch_dim][:,i] = λ_uniform
			λ_dict["Row_Norm"][sketch_dim][:,i] = λ_norm
			λ_dict["Cosmo"][sketch_dim][:,i] = λ_cosmo
		end
	end
	@save "QPresultsNEWNEW.jld2" time_dict cost_dict norm_dict x_dict λ_dict
end