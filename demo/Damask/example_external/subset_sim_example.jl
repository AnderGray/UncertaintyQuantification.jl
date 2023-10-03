include("../../src/UncertaintyQuantification.jl")
using UncertaintyQuantification, DelimitedFiles, Formatting


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


# Read output file and compute maximum (absolute) displacement
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
    return 9 .- reduce(vcat, df.y)
end

simulation_method_0 = FORM()
simulation_method_1 = MonteCarlo(500)
simulation_method_2 = SubSetSimulation(1000, 0.1, 10, Normal())
simulation_method_3 = SubSetInfinity(1000, 0.1, 10, 0.5)

@time pf_0, cov_0, samples_0 = probability_of_failure([ext], limitstate, inputs, simulation_method_0)

@time pf_1, cov_1, samples_1 = probability_of_failure([ext], limitstate, inputs, simulation_method_1)

@time pf_2, cov_2, samples_2 = probability_of_failure([ext], limitstate, inputs, simulation_method_2)

@time pf_3, cov_3, samples_3 = probability_of_failure([ext], limitstate, inputs, simulation_method_3)



# Probability of failure MC (2000)   : 0.0045
# Probability of failure SS (200 * 3): 0.004450000000000001

println("Probability of failure: $pf")
println("COV: $cov")

file = open("probability_of_failure_mc.txt", "w")
write(file, "pf = $pf \n cov = $cov")
close(file)

fid = h5open("result_mc", "w")
create_group(fid, "leakage")
fid["pf"] = pf
fid["cov"] = cov
fid["samples"] = samples.leakage
fid["inputs"] = Matrix(samples[!,["X1", "X2"]])

close(fid)

# rmprocs(workers()) # release the local workers
