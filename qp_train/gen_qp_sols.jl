using JLD2, LinearAlgebra
include("training_functions.jl")

n = 200
nc = 4000
sketch_dim = 400
sketch_type = :Row_Norm # :Uniform, :Row_Norm, :Full
A = randn(nc, n)
z = randn(n, 1)
b = vec(A * z + rand(nc))

R = randn(n, n)
Q = R' * R + 0.01 * I
#q = randn(n)
q = zeros(n)

@time x0, rc = cheby_lp(A, b, ones(length(b)), :Uniform)

@time x_ours_not, λ_ours_not = newton_sketch(Q, q, A, b, sketch_dim, :Full, x0, rc, tol=1e-4, μ=10)
@time x_ours, λ_ours = newton_sketch(Q, q, A, b, sketch_dim, sketch_type, x0, rc, tol=1e-4, μ=10)
# @time x_ipopt, λ_ipopt = solve_IPOPT(Q, q, A, b)
@time x_cosmo, λ_cosmo = solve_COSMO2(Q, q, A, b, x0)



function gen_data(n, nc)
    A = randn(nc, n)
    z = randn(n, 1)
    b = vec(A * z + rand(nc))
    
    R = randn(n, n)
    Q = R' * R + 0.01 * I

    return Q, A, b
end