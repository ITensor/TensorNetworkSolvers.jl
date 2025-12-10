using TensorNetworkSolvers: TensorNetworkSolvers
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
    TensorNetworkSolvers, :DocTestSetup, :(using TensorNetworkSolvers); recursive = true
)

include("make_index.jl")

makedocs(;
    modules = [TensorNetworkSolvers],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "TensorNetworkSolvers.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/TensorNetworkSolvers.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"],
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
    repo = "github.com/ITensor/TensorNetworkSolvers.jl", devbranch = "main", push_preview = true
)
