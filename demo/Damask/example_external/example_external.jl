###
#   Tutorial on how to use an external python script as a model
###
using UncertaintyQuantification, DelimitedFiles, Formatting, Distributions, Plots


X1 = RandomVariable(Normal(0, 1), :X1)
X2 = RandomVariable(Normal(0, 1), :X2)

inputs = [X1, X2]

# Source/Extra files are expected to be in this folder
sourcedir = joinpath(pwd())

# These files will be rendere through Mustach.jl and have values injected
sourcefile = "run_model.py"

# Dictionary to map format Strings (Formatting.jl) to variables
numberformats = Dict(:* => ".4e")

# UQ will create subfolders in here to run the solver and store the results
workdir = joinpath(pwd(), "run_dir")


# Read output file and get quantity of interest
# An extractor is based the working directory for the current sample
output = Extractor(base -> begin
    file = joinpath(base, "simulation.out")
    data = readdlm(file, ' ')

    return data[1]
end, :y)

solver = Solver(
    "python3", # How to call simulation
    "run_model.py";     # Simulation input file, to interpolate values into
    args="", # (optional) extra arguments passed to the solver
)

ext = ExternalModel(
    sourcedir, sourcefile, output, solver; workdir=workdir, formats=numberformats
)

function limitstate(df)
    return 4 .- reduce(vcat, df.y)
end


###
#   True failure for this problem
###
True_failure_prob = 1 - cdf(Normal(), 4/sqrt(2))

###
#   Simulation methods
###

simulation_method_1 = MonteCarlo(1000)
simulation_method_2 = SubSetSimulation(100, 0.1, 10, Normal())
simulation_method_3 = SubSetInfinity(100, 0.1, 10, 0.5)

@time pf_1, cov_1, samples_1 = probability_of_failure([ext], limitstate, inputs, simulation_method_1)

@time pf_2, cov_2, samples_2 = probability_of_failure([ext], limitstate, inputs, simulation_method_2)

@time pf_3, cov_3, samples_3 = probability_of_failure([ext], limitstate, inputs, simulation_method_3)



###
#   Print results
###
println()
println("###########################################################")
println("##### True failure probability: $True_failure_prob #####")
println("###########################################################")

alpha = 0.95    # Alpha confidence interval on computed probability

std_MC = cov_1 * pf_1
conf_lo_mc = quantile(Normal(pf_1, std_MC), (1-alpha)/2)
conf_hi_mc = quantile(Normal(pf_1, std_MC), 1 - (1-alpha)/2)
conf_lo_mc = max(conf_lo_mc, 0)

std_2 = cov_2 * pf_2
conf_lo_2 = quantile(Normal(pf_2, std_2), (1-alpha)/2)
conf_hi_2 = quantile(Normal(pf_2, std_2), 1 - (1-alpha)/2)
conf_lo_2 = max(conf_lo_2, 0)

std_3 = cov_3 * pf_3
conf_lo_3 = quantile(Normal(pf_3, std_3), (1-alpha)/2)
conf_hi_3 = quantile(Normal(pf_3, std_3), 1 - (1-alpha)/2)
conf_lo_3 = max(conf_lo_3, 0)

println("Monte Carlo: pf = $pf_1")
println("Nsamples = $(length(samples_1.y))")
println("$(100 * alpha) confidence interval: [$conf_lo_mc, $conf_hi_mc]")
println("###########################################################")

println("Subset 1:  pf = $pf_2")
println("Nsamples = $(length(samples_2.y))")
println("$(100 * alpha) confidence interval: [$conf_lo_2, $conf_hi_2]")
println("###########################################################")

println("Subset 2:  pf = $pf_3")
println("Nsamples = $(length(samples_3.y))")
println("$(100 * alpha) confidence interval: [$conf_lo_3, $conf_hi_3]")
println("###########################################################")


###
#   Plot results
###

# Monte Carlo
failures_MC = limitstate(samples_1) .<= 0

scatter(samples_1.X1, samples_1.X2, color = :blue)
scatter!(samples_1.X1[failures_MC], samples_1.X2[failures_MC], color = :red, label = "Failures")
xlabel!("X1")
ylabel!("X2")
title!("Input space MC")
savefig("inputs_mc.png")

histogram(samples_1.y)
histogram!(samples_1.y[failures_MC], color = :red, label = "Failures")
xlabel!("y")
savefig("outputs_mc.png")

# Subset 1

N_levels_ss_1 = maximum(samples_2.level)
scatter()
for i = 1:N_levels_ss_1
    level_samples = samples_2.level .== i
    scatter!(samples_2.X1[level_samples], samples_2.X2[level_samples], label = "level $i")
end
xlabel!("X1")
ylabel!("X2")
title!("Input space SS 1")
savefig("input_ss_1.png")


histogram()
for i = 1:N_levels_ss_1
    level_samples = samples_2.level .== i
    histogram!(samples_2.y[level_samples], label = "level $i")
end
xlabel!("y")
title!("Output space SS 1")
savefig("output_ss_1.png")

# Subset 2

N_levels_ss_2 = maximum(samples_3.level)
scatter()
for i = 1:N_levels_ss_2
    level_samples = samples_3.level .== i
    scatter!(samples_3.X1[level_samples], samples_3.X2[level_samples], label = "level $i")
end
xlabel!("X1")
ylabel!("X2")
title!("Input space SS 2")
savefig("input_ss_2.png")

histogram()
for i = 1:N_levels_ss_2
    level_samples = samples_3.level .== i
    histogram!(samples_3.y[level_samples], label = "level $i")
end
xlabel!("y")
title!("Output space SS 2")
savefig("output_ss_2.png")
