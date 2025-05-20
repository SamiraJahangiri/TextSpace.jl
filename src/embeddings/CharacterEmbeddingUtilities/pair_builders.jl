using Random

include("windowify.jl")
include("skipgram_pairs.jl")
include("cbow_pairs.jl")       #all helpers loaded

"""
    build_char_pairs(idstream; mode=:skipgram, win=128, stride=64, radius=5, rng=GLOBAL_RNG)

Slice an id stream into windows and return shuffled training pairs.
`mode` chooses the objective:

* `:skipgram` - `(center, context)` pairs
* `:cbow`     - `(context_vector, target)` pairs
"""
function build_char_pairs(ids::Vector{Int};
                     mode::Symbol = :skipgram,
                     win::Int     = 128,
                     stride::Int  = 64,
                     radius::Int  = 5,
                     rng          = Random.GLOBAL_RNG)

    wins = windowify(ids, win, stride)

    pairfn = mode === :skipgram ? w -> skipgram_pairs(w, radius) :
             mode === :cbow     ? w -> cbow_pairs(w, radius)     :
             error("Unknown mode $(mode). Choose :skipgram or :cbow")

    pairs = reduce(vcat, (pairfn(w) for w in wins))
    Random.shuffle!(rng, pairs)
    
    centers  = first.(pairs)    # Vector{Int}
    contexts = last.(pairs)     # Vector{Int}
    return centers, contexts
end
