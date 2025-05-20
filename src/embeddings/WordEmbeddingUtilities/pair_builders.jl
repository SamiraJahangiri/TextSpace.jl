using Random

include("windowify.jl")
include("skipgram_pairs.jl")
include("cbow_pairs.jl")


function build_word_pairs(ids::Vector{Int};
                          mode::Symbol = :skipgram,
                          win::Int = 11,
                          stride::Int = 11,
                          radius::Int = 5,
                          rng = Random.GLOBAL_RNG)

    wins = windowify(ids; win=win, stride=stride)

    pairfn = mode === :skipgram ? w -> skipgram_pairs(w, radius) :
             mode === :cbow     ? w -> cbow_pairs(w, radius)     :
             error("mode must be :skipgram or :cbow")

    pairs = reduce(vcat, (pairfn(w) for w in wins))
    Random.shuffle!(rng, pairs)

    centres  = first.(pairs)
    contexts = last.(pairs)
    return centres, contexts
end
