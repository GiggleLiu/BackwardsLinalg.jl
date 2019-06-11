using Documenter, BackwardsLinalg

makedocs(;
    modules=[BackwardsLinalg],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/{commit}{path}#L{line}",
    sitename="BackwardsLinalg.jl",
    authors="Jin-Guo Liu, Lei Wang",
    assets=[],
)

deploydocs(;
    repo="github.com/GiggleLiu/BackwardsLinalg.jl",
)
