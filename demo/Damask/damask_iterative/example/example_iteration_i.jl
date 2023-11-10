using UncertaintyQuantification, DelimitedFiles, Formatting
using Plots

###
#   Run iteration i-1 first
#   Required. Samples from previous iteration, and evaluated proposal samples
###

sim = SubSetInfinity(1000, 0.1, 10, 0.5)

function accept_reject(next_samples, prev_samples, threshold, performancefunction)

    number_of_seeds = Int64(max(1, ceil(sim.n * sim.target)))       # Number a samples which 'make it'
    samples_per_seed = Int64(floor(sim.n / number_of_seeds))        

    presamplesperformance = performancefunction(prev_samples)
    nextlevelperformance = performancefunction(next_samples)

    sorted_indices = sortperm(presamplesperformance)

    prev_samples = prev_samples[sorted_indices,:]

    pre_samples_seeds = prev_samples[1:number_of_seeds,:]
    pre_samples_seeds = repeat(pre_samples_seeds, samples_per_seed)

    reject = nextlevelperformance .> threshold

    @info "rejection rate" mean(reject)

    next_samples[reject, :] = pre_samples_seeds[reject, :]

    return next_samples
end


function iteration_i(samples, sim, performancefunction, i, pf, threshold, cov)

    performance = [performancefunction(samples[end])]               # Evaluate performance function

    number_of_seeds = Int64(max(1, ceil(sim.n * sim.target)))       # Number a samples which 'make it'
    samples_per_seed = Int64(floor(sim.n / number_of_seeds))        

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
        println("Final level reached")
        println("Pf = $(prod(pf))")
        return samples[end], threshold, pf, cov
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


i += 1

evaluate!(model, next_samples)                                  # Run model

next_samples = accept_reject(next_samples, samples[i-1], threshold[i-1], performancefunction)
push!(samples, next_samples)

next_samples, threshold, pf, cov = iteration_i(samples, sim, performancefunction, i, pf, threshold, cov)


# scatter!(samples[i].X1, samples[i].X2)
scatter!(next_samples.X1, next_samples.X2)
