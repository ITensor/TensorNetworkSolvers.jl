# # TensorNetworkSolvers.jl
#
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/TensorNetworkSolvers.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/TensorNetworkSolvers.jl/dev/)
# [![Build Status](https://github.com/ITensor/TensorNetworkSolvers.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/TensorNetworkSolvers.jl/actions/workflows/Tests.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/ITensor/TensorNetworkSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/TensorNetworkSolvers.jl)
# [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
# [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# ## Support
#
# {CCQ_LOGO}
#
# TensorNetworkSolvers.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

# ## Installation instructions

# This package resides in the `ITensor/ITensorRegistry` local registry.
# In order to install, simply add that registry through your package manager.
# This step is only required once.
#=
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url = "https://github.com/ITensor/ITensorRegistry")
```
=#
# or:
#=
```julia
julia> Pkg.Registry.add(url = "git@github.com:ITensor/ITensorRegistry.git")
```
=#
# if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

# Then, the package can be added as usual through the package manager:

#=
```julia
julia> Pkg.add("TensorNetworkSolvers")
```
=#

# ## Examples

# Perform a single sweep.
using Graphs: path_graph
using TensorNetworkSolvers: dmrg, dmrg_sweep
operator = path_graph(4)
regions = [(1, 2), (2, 3), (3, 4)]
tol = 1.0e-4
maxdim = 50
region_kwargs = (;
    update = (; tol),
    insert = (; maxdim),
)
state = []
x1 = dmrg_sweep(operator, state; regions, region_kwargs)

# Sweep-dependent region kwargs (uniform across regions).
using Graphs: path_graph
using TensorNetworkSolvers: dmrg, dmrg_sweep
operator = path_graph(4)
regions = [(1, 2), (2, 3), (3, 4)]
nsweeps = 3
tols = [1.0e-3, 1.0e-4, 1.0e-5]
maxdims = [20, 50, 100]
region_kwargs = map(1:nsweeps) do i
    return (;
        update = (; tol = tols[i]),
        insert = (; maxdim = maxdims[i]),
    )
end
state = []
x2 = dmrg(operator, state; nsweeps, regions, region_kwargs)

# Region-dependent kwargs.
using Graphs: path_graph
using TensorNetworkSolvers: dmrg, dmrg_sweep
operator = path_graph(4)
regions = [(1, 2), (2, 3), (3, 4)]
nsweeps = 3
tols = [1.0e-3, 1.0e-4, 1.0e-5]
maxdims = [20, 50, 100]
region_kwargs = map(1:nsweeps) do i
    return function (problem, alg, region_state)
        return (;
            update = (; tol = tols[i] / region_state.iteration),
            insert = (; maxdim = maxdims[i] * region_state.iteration),
        )
    end
end
state = []
x3 = dmrg(operator, state; nsweeps, regions, region_kwargs)
