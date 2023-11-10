using UncertaintyQuantification, DelimitedFiles, Formatting
using Plots

###
#   Define input random variables
###
X1 = RandomVariable(Uniform(-4, 4), :X1)
X2 = RandomVariable(Uniform(-4, 4), :X2)

inputs = [X1, X2]

###
#   Define function, and model
###
function y(df)
    return df.X1 .+ df.X2
end

model = Model(y, :y)

###
#   Define limitstate. Failure is limitstate <= 0
###
function performancefunction(df)
    return 7.9 .- reduce(vcat, df.y)
end

###
#   True failure for this problem
###
True_failure_prob = 1 - cdf(Normal(), 9/sqrt(2))

###
#   Define simulation parameters
#   inputs
#       1. Number of samples
#       2. Target intermidaite failure probability
#       3. Max number of levels
#       4. Standard deviation of proposal distribution
###
sim = SubSetInfinity(1000, 0.1, 10, 0.5)

###
#   Read samples and evaluation from text file, and write to a data Frame
###
samples = [UncertaintyQuantification.sample(inputs, sim)]       # Sample inputs
evaluate!(model, samples[end])                                  # Run model



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

    to_physical_space!(inputs, nextlevelsamples)

    @info "Subset level $i sampling"  pf[i]  threshold[i] cov[i]

    return nextlevelsamples, threshold, pf, cov
end

i = 1

next_samples, threshold, pf, cov = iteration_zero_and_1(samples, sim, performancefunction)

scatter(samples[1].X1, samples[1].X2)
scatter!(next_samples.X1, next_samples.X2)
