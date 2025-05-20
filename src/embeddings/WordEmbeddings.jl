module WordEmbeddings

using Flux, Random
using Flux: softplus     
using Statistics: mean
using Random: randn
using LinearAlgebra: dot, norm
using Zygote: Buffer


include(joinpath(@__DIR__, "WordEmbeddingUtilities", "__init__.jl"))
using .WordEmbeddingUtilities: build_word_pairs

export train!, embeddings, save_embeddings, load_embeddings, vector,
       SkipGramModel 


# convenience alias for the vocabulary type defined in Preprocessing.jl
using ..Preprocessing: Vocabulary    
const V = Vocabulary


#model definition
struct SkipGramModel
    emb :: Flux.Embedding               # (emb_dim × vocab_size) weights
end
Flux.@layer SkipGramModel

SkipGramModel(vocab_size::Int, emb_dim::Int) = SkipGramModel(
    Flux.Embedding(randn(Float32, emb_dim, vocab_size) .* 0.01f0)
)

embeddings(m::SkipGramModel) = cpu(m.emb.weight)              # matrix view

sigm(x) = 1f0 ./ (1f0 .+ exp.(-x))

dot_scores(m::SkipGramModel, ctrs, ctxs) =
    vec(sum(m.emb(ctrs) .* m.emb(ctxs); dims = 1))

sg_loss(m, pc, po, nc, no) =
    mean(vcat( softplus.(-dot_scores(m, pc, po)),     # log σ(pos)
               softplus.( dot_scores(m, nc, no)) ))   # log σ(-neg)

# CBOW
function cbow_scores(m::SkipGramModel,
                     ctxs::Vector{<:AbstractVector{Int}},
                     ctrs::Vector{Int})

    @assert length(ctxs) == length(ctrs)
    n   = length(ctrs)
    buf = Buffer(Vector{Float32}(undef, n))      # Zygote-friendly scratch

    W = embeddings(m)                            # emb_dim × |V|

    @inbounds for k in 1:n
        μ_ctx = mean(W[:, ctxs[k]]; dims = 2)[:, 1]   # emb_dim vector
        buf[k] = dot(μ_ctx, @view W[:, ctrs[k]])      # write into Buffer
    end
    return copy(buf)                              # ordinary tracked Vector
end

cbow_loss(m, pcx, pctr, ncx, nctr) =
    mean(vcat( softplus.(-cbow_scores(m, pcx, pctr)),
               softplus.( cbow_scores(m, ncx, nctr)) ))


               
@inline function repeat_vec(src::AbstractVector{T}, k::Int) where T
    dest = Vector{T}(undef, length(src)*k)
    @inbounds for t in 0:k-1, i in eachindex(src)
        dest[t*length(src)+i] = src[i]
    end
    return dest
end


# Trainer (Skip-Gram + CBOW, negative sampling)
function train!(ids::Vector{Int}, vocab::V;
                objective::Symbol = :skipgram,
                emb_dim::Int      = 128,
                radius::Int       = 5,
                epochs::Int       = 5,
                batch::Int        = 1024,
                k_neg::Int        = 5,
                lr                = 1e-2,
                rng               = Random.GLOBAL_RNG)

    @assert objective ∈ (:skipgram, :cbow)

    p1, p2 = build_word_pairs(ids; mode = objective,
                                      radius = radius, rng = rng)

    centres, contexts, lossfun = objective === :skipgram ?
        (p1, p2, sg_loss) :                      # (ctr , ctx)
        (p2, p1, cbow_loss)                      # (ctr , ctx) for CBOW

    N          = length(centres)
    vocab_size = length(vocab.id2token)

    model = SkipGramModel(vocab_size, emb_dim)
    opt   = Flux.Adam(lr)

    for epoch in 1:epochs
        for i in 1:batch:N
            j = min(i+batch-1, N)

            if objective === :skipgram
                pc  = centres[i:j]
                po  = contexts[i:j]
                nc  = repeat_vec(pc, k_neg)
                no  = rand(rng, 1:vocab_size, length(po)*k_neg)

                gs = gradient(() -> lossfun(model, pc, po, nc, no),
                              Flux.params(model))
            else
                pcx  = contexts[i:j]                     # Vec{Vec{Int}}
                pctr = centres[i:j]
                ncx  = vcat(Iterators.repeated(pcx, k_neg)...)  # cheap concat
                nctr = rand(rng, 1:vocab_size, length(pctr)*k_neg)

                gs = gradient(() -> lossfun(model, pcx, pctr, ncx, nctr),
                              Flux.params(model))
            end
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        @info "epoch $epoch finished"
    end
    return model
end



"""
    save_embeddings(path, model, vocab)

Write a TSV: `word \\t f1 \\t f2 ...` (one row per token).
"""
function save_embeddings(path, m::SkipGramModel, vocab::V)
    open(path, "w") do io
        for (tok, vec) in zip(vocab.id2token, eachcol(embeddings(m)))
            println(io, join((tok, vec...), '\t'))
        end
    end
end

"""
    model, vocab = load_embeddings(path)

Load a TSV created by `save_embeddings` and rebuild a *frozen* model.
"""
function load_embeddings(path::AbstractString)
    toks  = String[]
    cols  = Vector{Float32}[]

    for line in eachline(path)
        parts = split(line, '\t')
        push!(toks, parts[1])
        push!(cols, Float32.(parse.(Float64, parts[2:end])))
    end

    W = hcat(cols...)                         # emb_dim × |V|
    vocab = V(Dict(tok=>i for (i,tok) in enumerate(toks)),
              toks, Dict{Int,Int}(), 
              findfirst(isequal("<unk>"), toks))

    m = SkipGramModel(size(W,2), size(W,1))
    m.emb.weight .= W                         # copy weights
    return m, vocab
end

"""
    vec = vector(model, vocab, token)

Return the embedding for a word (String) or character (Char).
Falls back to `<unk>` when unseen.
"""
function vector(m::SkipGramModel, vocab::V, tok::AbstractString)
    id = get(vocab.token2id, tok, vocab.unk_id)
    return @view embeddings(m)[:, id]
end
vector(m::SkipGramModel, vocab::V, c::Char) = vector(m, vocab, string(c))




end # module