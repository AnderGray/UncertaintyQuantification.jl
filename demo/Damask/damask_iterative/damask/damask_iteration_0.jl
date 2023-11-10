using UncertaintyQuantification, DelimitedFiles, Formatting
using Plots

###
#   Define input random variables
###
# Num_dim = 2005
Num_dim = 120

for j = 1:Num_dim
    eval(:($(Symbol(:X,j)) = RandomVariable(Uniform(0, 1), (Symbol(:X,$j))) ))
end

inputs = [eval(:($(Symbol(:X,i)))) for i = 1:Num_dim]

###
#   Dummy model
###
function y(df)
    return df.X1 .+ df.X2
end

model = Model(y, :y)

###
#   Define limitstate. Failure is limitstate <= 0
###
function performancefunction(df)
    return reduce(vcat, df.y) .- 55e6
end


###
#   Define simulation parameters
#   inputs
#       1. Number of samples
#       2. Target intermidaite failure probability
#       3. Max number of levels
#       4. Standard deviation of proposal distribution
###
sim = SubSetInfinity(100, 0.1, 10, 0.5)

###
#   Generate a DataFrame using toy model, read samples and damask evaluation from text file, 
#   and write to data Frame
###
samples = [UncertaintyQuantification.sample(inputs, sim)]       # Sample inputs
evaluate!(model, samples[end])                                  # Run model

input_data = readdlm("sim_data/sim_0/input_0.txt")
output_data = readdlm("sim_data/sim_0/yield_stresses_0.txt")

samples[1].y .= output_data
samples[1][!, 1:120] = input_data


function iteration_zero_and_1(samples, sim, performancefunction)

    performance = [performancefunction(samples[end])]               # Evaluate performance function

    number_of_seeds = Int64(max(1, ceil(sim.n * sim.target)))       # Number a samples which 'make it'
    samples_per_seed = Int64(floor(sim.n / number_of_seeds))        

    threshold = zeros(sim.levels, 1)                                # Intermediate performance functions
    pf = ones(sim.levels, 1)
    cov = zeros(sim.levels, 1)

    ## Iteration 1
    i = 1

    sorted_performance = sort(performance[end])
    sorted_indices = sortperm(performance[end])

    ###
    # Threshold defined as output 'sim.target' (0.1) quantile
    ###
    threshold[i] = sorted_performance[number_of_seeds]              

    
    ###
    # Evaluate intermediate probability. Expected value of number of samples in intermediate failure
    # If final level, pf = mean(performance[end] .<= 0)
    ###
    pf[i] = if threshold[i] <= 0                                    
        mean(performance[end] .<= 0)
    else
        mean(performance[end] .<= threshold[i])
    end

    ###
    # Estimate Cov
    ###
    if i == 1                                                       
        cov[i] = sqrt((pf[i] - pf[i]^2) / sim.n) / pf[i]
        ## MarkovChain coefficient of variation
    else
        Iᵢ = reshape(
            performance[end] .<= max(threshold[i], 0), number_of_seeds, samples_per_seed
        )
        cov[i] = UncertaintyQuantification.estimate_cov(Iᵢ, pf[i], sim.n)
    end

    if threshold[i] <= 0 || i == sim.levels
        println("ERROR: final level reached in first iteration")
        return samples, threshold, pf, cov
    end

    ###
    # Sample next samples from seeds and proposal distribution
    ###
    samples_level = samples[end][sorted_indices[1:number_of_seeds], :]
    performance_level = sorted_performance[1:number_of_seeds]

    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    rvs = names(random_inputs)

    UncertaintyQuantification.to_standard_normal_space!(inputs, samples_level)

    samples_level = repeat(samples_level, samples_per_seed)
    performance_level = repeat(performance_level, samples_per_seed)

    means = Matrix{Float64}(samples_level[:, rvs]) .* sqrt(1 - sim.s^2)

    nextlevelsamples = copy(samples_level)
    nextlevelsamples[:, rvs] = randn(size(means)) .* sim.s .+ means

    UncertaintyQuantification.to_physical_space!(inputs, nextlevelsamples)

    @info "Subset level $i sampling"  pf[i]  threshold[i] cov[i]

    return nextlevelsamples, threshold, pf, cov
end

next_samples, threshold, pf, cov = iteration_zero_and_1(samples, sim, performancefunction)

###
#   Save output to Datafile 
###

mkpath("sim_data/sim_1")

open("sim_data/sim_1/input_1.txt", "w") do io
    writedlm(io, Matrix(next_samples[!, 1:120]))
end

open("sim_data/sim_0/threshold_0.txt", "w") do io
    writedlm(io, threshold)
end


open("sim_data/sim_0/pf_0.txt", "w") do io
    writedlm(io, pf)
end

open("sim_data/sim_0/cov_0.txt", "w") do io
    writedlm(io, cov)
end


###
#
###
scatter(samples[1].X70, samples[1].X80)
scatter!(next_samples.X70, next_samples.X80)


histogram(samples[1].X70, bins = 30)
histogram!(next_samples.X70, bins = 30)


histogram(samples[1].y, bins = 30)