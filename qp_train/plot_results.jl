using JLD2, FileIO, Statistics, Plots, LinearAlgebra
pyplot()
close("all")

function rel_norm_mats(A, B)
    # Computes the ||A[:, i] - B[:, i]|| / ||B[:, i]|| ∀ i in 1:size(A or B, 2)
    diff = A - B
    ndiff = [norm(diff[:, i]) for i in 1:size(A, 2)]
    nB = [norm(B[:, i]) for i in 1:size(A, 2)]
    return ndiff ./ nB
end
data = load("QPresultsNEW.jld2")

td = data["time_dict"]
cod= data["cost_dict"]
nd = data["norm_dict"]
ld = data["λ_dict"]
xd = data["x_dict"]

base_method = "Full"
sketch_dims = collect(keys(td["Full"]))
sort!(sketch_dims)
niters = size(td["Full"][sketch_dims[1]], 1)

mean_times = Matrix{Float64}(undef, length(sketch_dims), 3) # [uniform row_norm full(not used)]
std_times = Matrix{Float64}(undef, length(sketch_dims), 3)

mean_rcosts = Matrix{Float64}(undef, length(sketch_dims), 3) # [uniform row_norm]
std_rcosts = Matrix{Float64}(undef, length(sketch_dims), 3)

mean_gnorms = Matrix{Float64}(undef, length(sketch_dims), 3) # [uniform row_norm]
std_gnorms = Matrix{Float64}(undef, length(sketch_dims), 3)

mean_rxnorms = Matrix{Float64}(undef, length(sketch_dims), 3)
std_rxnorms = Matrix{Float64}(undef, length(sketch_dims), 3)

mean_rlnorms = Matrix{Float64}(undef, length(sketch_dims), 3)
std_rlnorms = Matrix{Float64}(undef, length(sketch_dims), 3)

all_times_full = Vector{Float64}()
all_times_cosmo = Vector{Float64}()


for (i, sdim) in enumerate(sketch_dims)
    append!(all_times_full, td["Full"][sdim])
    append!(all_times_cosmo, td["Cosmo"][sdim])
    
    for (j, method) in enumerate(["Uniform", "Row_Norm", "Full"])
        mean_times[i, j] = mean(td[method][sdim])
        std_times[i, j] = std(td[method][sdim])

        mean_gnorms[i, j] = mean(nd[method][sdim])
        std_gnorms[i, j] = std(nd[method][sdim])
        
        mean_rcosts[i, j] = mean(abs.(cod[method][sdim] - cod[base_method][sdim]) ./ abs.(cod[base_method][sdim]))
        std_rcosts[i, j] = std(abs.(cod[method][sdim] - cod[base_method][sdim]) ./ abs.(cod[base_method][sdim]))
        
        mean_rxnorms[i, j] = mean(rel_norm_mats(xd[method][sdim], xd[base_method][sdim]))
        std_rxnorms[i, j] = std(rel_norm_mats(xd[method][sdim], xd[base_method][sdim]))

        mean_rlnorms[i, j] = mean(rel_norm_mats(ld[method][sdim], ld[base_method][sdim]))
        std_rlnorms[i, j] = std(rel_norm_mats(ld[method][sdim], ld[base_method][sdim]))
    end
end

mean_times_full = mean(all_times_full) * ones(length(sketch_dims))
mean_times_cosmo = mean(all_times_cosmo) * ones(length(sketch_dims))
std_times_full = std(all_times_full) * ones(length(sketch_dims))
std_times_cosmo = std(all_times_cosmo) * ones(length(sketch_dims))

tplt = plot(sketch_dims, mean_times_full, ribbon=std_times_full, fillalpha=0.2, label="Full")
plot!(tplt, sketch_dims, mean_times[:, 1], ribbon=std_times[:, 1], fillalpha=0.2, label="Uniform")
plot!(tplt, sketch_dims, mean_times[:, 2], ribbon=std_times[:, 2], fillalpha=0.2, label="Row Norm")
#plot!(tplt, sketch_dims, mean_times_cosmo, ribbon=std_times_cosmo, fillalpha=0.2, label="Cosmo")
title!("Timing")
xaxis!("Sketch Dimension")
yaxis!("Time (s)")
display(tplt)
savefig(tplt, "timing.png")

cplt = plot(sketch_dims, mean_rcosts[:, 1], ribbon=std_rcosts[:, 1], fillalpha=0.2, label="Uniform")
plot!(cplt, sketch_dims, mean_rcosts[:, 2], ribbon=std_rcosts[:, 2], fillalpha=0.2, label="Row Norm")
title!("Relative Cost wrt No Sketching")
xaxis!("Sketch Dimension")
yaxis!("Relative Cost")
display(cplt)
savefig(cplt, "cost.png")

ngplt = plot(sketch_dims, mean_gnorms[:, 1], ribbon=std_gnorms[:, 1], fillalpha=0.2, label="Uniform")
plot!(ngplt, sketch_dims, mean_gnorms[:, 2], ribbon=std_gnorms[:, 2], fillalpha=0.2, label="Row Norm")
title!("Norm Diff of Gradients")
xaxis!("Sketch Dimension")
yaxis!("Norm Diff of Gradients")
display(ngplt)
savefig(ngplt, "grad_norm.png")

nxplt = plot(sketch_dims, mean_rxnorms[:, 1], ribbon=std_rxnorms[:, 1], fillalpha=0.2, label="Uniform")
plot!(nxplt, sketch_dims, mean_rxnorms[:, 2], ribbon=std_rxnorms[:, 2], fillalpha=0.2, label="Row Norm")
title!("Relative Error of Primal Variable wrt No Sketching")
xaxis!("Sketch Dimension")
yaxis!("Relative Primal Error")
display(nxplt)
savefig(nxplt, "primal_norm.png")

@show mean_rlnorms
nlplt = plot(sketch_dims, mean_rlnorms[:, 1], ribbon=std_rlnorms[:, 1], fillalpha=0.2, label="Uniform")
plot!(nlplt, sketch_dims, mean_rlnorms[:, 2], ribbon=std_rlnorms[:, 2], fillalpha=0.2, label="Row Norm")
title!("Relative Error of Dual Variable wrt No Sketching")
xaxis!("Sketch Dimension")
yaxis!("Relative Dual Error")
display(nlplt)
savefig(nlplt, "dual_norm.png")
