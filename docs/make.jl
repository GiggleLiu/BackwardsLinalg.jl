using Documenter, LinalgBackwards

makedocs(;
    modules=[LinalgBackwards],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/GiggleLiu/LinalgBackwards.jl/blob/{commit}{path}#L{line}",
    sitename="LinalgBackwards.jl",
    authors="Jin-Guo Liu, Lei Wang",
    assets=[],
)

deploydocs(;
    repo="github.com/GiggleLiu/LinalgBackwards.jl",
)
