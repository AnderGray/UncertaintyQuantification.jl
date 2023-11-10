using UncertaintyQuantification, DelimitedFiles, Formatting
using Plots


##
# Choose iteration level
i = 2

mkpath("sim_data/sim_$i")

###
#   Define input random variables
###
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

prev_samples = UncertaintyQuantification.sample(inputs, sim)           # Sample inputs
proposal_samples = UncertaintyQuantification.sample(inputs, sim)       # Sample inputs
evaluate!(model, prev_samples)  # Make dummy dataframe
evaluate!(model, proposal_samples)  # Make dummy dataframe

prev_input_data = readdlm("sim_data/sim_$(i-2)/input_$(i-2).txt")
prev_yield = readdlm("sim_data/sim_$(i-2)/yield_stresses_$(i-2).txt")

prev_samples.y .= prev_yield
prev_samples[!, 1:120] .= prev_input_data

proposal_input_data = readdlm("sim_data/sim_$(i-1)/input_$(i-1).txt")
proposal_yield = readdlm("sim_data/sim_$(i-1)/yield_stresses_$(i-1).txt")

proposal_samples.y .= proposal_yield
proposal_samples[!, 1:120] .= proposal_input_data

pf = readdlm("sim_data/sim_$(i-2)/pf_$(i-2).txt")
threshold = readdlm("sim_data/sim_$(i-2)/threshold_$(i-2).txt")
cov = readdlm("sim_data/sim_$(i-2)/cov_$(i-2).txt")

function accept_reject(proposal_samples, prev_samples, threshold, performancefunction)

    number_of_seeds = Int64(max(1, ceil(sim.n * sim.target)))       # Number a samples which 'make it'
    samples_per_seed = Int64(floor(sim.n / number_of_seeds))        

    presamplesperformance = performancefunction(prev_samples)
    nextlevelperformance = performancefunction(proposal_samples)

    sorted_indices = sortperm(presamplesperformance)

    pre_samples_seeds = prev_samples[sorted_indices,:]

    pre_samples_seeds = pre_samples_seeds[1:number_of_seeds,:]
    pre_samples_seeds = repeat(pre_samples_seeds, samples_per_seed)

    reject = nextlevelperformance .> threshold

    @info "rejection rate" mean(reject) 

    next_samples = deepcopy(proposal_samples)

    next_samples[reject, :] = pre_samples_seeds[reject, :]

    return next_samples
end


function iteration_i(samples, sim, performancefunction, i, pf, threshold, cov)

    performance = performancefunction(samples)               # Evaluate performance function

    number_of_seeds = Int64(max(1, ceil(sim.n * sim.target)))       # Number a samples which 'make it'
    samples_per_seed = Int64(floor(sim.n / number_of_seeds))        

    sorted_performance = sort(performance)
    sorted_indices = sortperm(performance)

    ###
    # Threshold defined as output 'sim.target' (0.1) quantile
    ###
    threshold[i] = sorted_performance[number_of_seeds]              

    
    ###
    # Evaluate intermediate probability. Expected value of number of samples in intermediate failure
    # If final level, pf = mean(performance[end] .<= 0)
    ###
    pf[i] = if threshold[i] <= 0                                    
        mean(performance .<= 0)
    else
        mean(performance .<= threshold[i])
    end

    ###
    # Estimate Cov
    ###
    if i == 1                                                       
        cov[i] = sqrt((pf[i] - pf[i]^2) / sim.n) / pf[i]
        ## MarkovChain coefficient of variation
    else
        Iᵢ = reshape(
            performance .<= max(threshold[i], 0), number_of_seeds, samples_per_seed
        )
        cov[i] = UncertaintyQuantification.estimate_cov(Iᵢ, pf[i], sim.n)
    end

    if threshold[i] <= 0 || i == sim.levels
        println("Final level reached")
        println("Pf = $(prod(pf))")
        return samples, threshold, pf, cov
    end

    ###
    # Sample next samples from seeds and proposal distribution
    ###
    samples_level = samples[sorted_indices[1:number_of_seeds], :]
    performance_level = sorted_performance[1:number_of_seeds]

    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    rvs = names(random_inputs)

    UncertaintyQuantification.to_standard_normal_space!(inputs, samples_level)

    samples_level = repeat(samples_level, samples_per_seed)
    performance_level = repeat(performance_level, samples_per_seed)

    means = Matrix{Float64}(samples_level[:, rvs]) .* sqrt(1 - sim.s^2)

    nextlevelsamples = copy(samples_level)
    nextlevelsamples[:, rvs] = randn(size(means)) .* sim.s .+ means

    to_physical_space!(inputs, nextlevelsamples)

    @info "Subset level $i sampling"  pf[i]  threshold[i] cov[i]

    return nextlevelsamples, threshold, pf, cov
end


samples_accept = accept_reject(proposal_samples, prev_samples, threshold[i-1], performancefunction)

histogram(prev_samples.y, bins = 30)
histogram!(proposal_samples.y, bins = 30)
histogram!(samples_accept.y, bins = 30)

###
#   Save the samples which made it.
###
open("sim_data/sim_$(i-1)/input_accept_$(i-1).txt", "w") do io
    writedlm(io, Matrix(samples_accept[!, 1:120]))
end

open("sim_data/sim_$(i-1)/yield_accept_$(i-1).txt", "w") do io
    writedlm(io, samples_accept.y)
end

next_proposals, threshold, pf, cov = iteration_i(samples_accept, sim, performancefunction, i, pf, threshold, cov)


open("sim_data/sim_$(i)/input_$i.txt", "w") do io
    writedlm(io, Matrix(next_proposals[!, 1:120]))
end

open("sim_data/sim_$(i-1)/threshold_$(i-1).txt", "w") do io
    writedlm(io, threshold)
end

open("sim_data/sim_$(i-1)/pf_$(i-1).txt", "w") do io
    writedlm(io, pf)
end

open("sim_data/sim_$(i-1)/cov_$(i-1).txt", "w") do io
    writedlm(io, cov)
end


histogram(prev_samples.y, bins = 30)
histogram!(next_proposals.y, bins = 30)
