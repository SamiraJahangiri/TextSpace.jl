module CharacterEmbeddings

using Flux, Random
using Flux: softplus
using Zygote: Buffer
using Statistics: mean 
using Random: randn


include(joinpath(@__DIR__, "CharacterEmbeddingUtilities", "__init__.jl"))
using .CharacterEmbeddingUtilities: build_char_pairs


export train!, embeddings, save_embeddings, load_embeddings, vector  


struct SkipGramModel
    emb :: Flux.Embedding          # (emb_dim, vocab_size)
end
Flux.@layer SkipGramModel


function SkipGramModel(vocab_size::Int, emb_dim::Int)
    w = randn(Float32, emb_dim, vocab_size) .* 0.01f0 #sigma≈0.01
    SkipGramModel(Flux.Embedding(w))
end

embeddings(m::SkipGramModel) = cpu(m.emb.weight)          # emb_dim times vocab

sigm(x) = 1 ./ (1 .+ exp.(-x))      

function sg_loss(m, pos_c, pos_o, neg_c, neg_o)
    pos = dot_scores(m, pos_c, pos_o)        # length = B
    neg = dot_scores(m, neg_c, neg_o)        # length = B*k
    return mean( vcat( softplus.(-pos),      # log σ(pos)
                       softplus.( neg) ) )   # log σ(-neg)
end

function cbow_loss(m, ctxs, ctrs, neg_ctxs, neg_ctrs)
    pos = cbow_scores(m, ctxs, ctrs)
    neg = cbow_scores(m, neg_ctxs, neg_ctrs)
    return mean( vcat( softplus.(-pos), softplus.(neg) ) )
end


function dot_scores(m::SkipGramModel, centers, contexts)
    z_c = m.emb(centers)
    z_o = m.emb(contexts)
    return vec(sum(z_c .* z_o; dims = 1))
end


#TODO: still needed?
function cbow_dot_scores(m::SkipGramModel,
    ctxs::Vector{<:AbstractVector{Int}},
    centres::Vector{Int})
    W   = embeddings(m)                    # emb_dim x |V|
    out = Vector{Float32}(undef, length(centres))
    @inbounds for (k, cid) in pairs(centres)
        ctx_vec = mean(W[:, ctxs[k]]; dims = 2)  # emb_dim × 1
        ctr_vec = @view W[:, cid]               # emb_dim
        out[k]  = sum(ctx_vec .* ctr_vec)
    end
    return out
end



function cbow_scores(m::SkipGramModel,
    ctxs::Vector{<:AbstractVector{Int}},
    centres::Vector{Int})

    @assert length(ctxs) == length(centres)
    n = length(centres)

    buf = Buffer(Vector{Float32}(undef, n))   # Zygote-approved scratch

    for k in 1:n
        # mean(context embeddings) - `mean` gives emb_dim×1 matrix
        μ_ctx   = @view mean(m.emb(ctxs[k]); dims = 2)[:, 1]
        centre  = m.emb(centres[k])           # emb_dim vector
        buf[k]  = sum(μ_ctx .* centre)        # dot-product
    end

    return copy(buf)                          # ordinary tracked Vector
end

function cbow_loss(m, pos_ctx, pos_ctr, neg_ctx, neg_ctr)
    l_pos = log.(sigm.(cbow_scores(m, pos_ctx, pos_ctr)))
    l_neg = log.(sigm.(-cbow_scores(m, neg_ctx, neg_ctr)))
    return -mean(vcat(l_pos, l_neg))
end


function train!(ids::Vector{Int}, vocab;
        objective::Symbol = :skipgram,
        emb_dim::Int = 128,
        radius::Int = 5,
        epochs::Int = 5,
        batch::Int = 1024,
        k_neg::Int = 5,
        lr = 1e-2,
        rng = Random.GLOBAL_RNG)

    @assert objective ∈ (:skipgram, :cbow)

    p1, p2   = build_char_pairs(ids; mode=objective,
                            radius=radius, rng=rng)
    if objective == :skipgram
        centres, contexts = p1, p2
        lossfun = sg_loss
    else
        contexts, centres = p1, p2    # CBOW order!
        lossfun = cbow_loss
    end

    N          = length(centres)
    vocab_size = length(vocab.id2token)
    model      = SkipGramModel(vocab_size, emb_dim)
    opt        = Flux.Adam(lr)

    for epoch in 1:epochs
        for i in 1:batch:N
            j = min(i+batch-1, N)

            if objective == :skipgram
                pc  = centres[i:j]
                po  = contexts[i:j]
                nc  = repeat(pc, k_neg)
                no  = rand(rng, 1:vocab_size, length(po)*k_neg)

                gs = gradient(() -> lossfun(model, pc, po, nc, no),
                            Flux.params(model))

            else      #  CBOW
                pcx  = contexts[i:j]                # Vector{Vector{Int}}
                pctr = centres[i:j]
                ncx  = vcat(Iterators.repeated(pcx, k_neg)...)  # copy refs OK
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
    save_embeddings
    
save a tsv file
"""
function save_embeddings(path, m::SkipGramModel, vocab)
    open(path, "w") do io
        for (tok, vec) in zip(vocab.id2token, eachcol(embeddings(m)))
            println(io, join((tok, vec...), '\t'))
        end
    end
end

"Return the embedding vector for the given character (or <unk> fallback)."
function vector(m::SkipGramModel, vocab, ch::AbstractString)
    id = get(vocab.token2id, ch, vocab.unk_id)
    return @view embeddings(m)[:, id]   # alias, no copy
end
vector(m::SkipGramModel, vocab, ch::Char) = vector(m, vocab, string(ch)) 



"""
    load_embeddings(path) -> (model, vocab)

Read a TSV file produced by [`save_embeddings`](@ref) and reconstruct an
*inference-only* SkipGram/CBOW model (same embedding matrix) plus its
`Vocabulary`.  The returned model can be queried with [`vector`](@ref) and
[`embeddings`](@ref) but **is not meant for further training**.
"""
function load_embeddings(path::AbstractString)
    tokens  = String[]
    columns = Vector{Float32}[]

    for line in eachline(path)
        parts = split(line, '\t')
        push!(tokens, parts[1])
        push!(columns, Float32.(parse.(Float64, parts[2:end])))
    end

    E = hcat(columns...)              # (emb_dim × |V|)
    vocab = V(Dict(tok=>i for (i,tok) in enumerate(tokens)),
              tokens,
              Dict{Int,Int}(),        # empty char-class map
              findfirst(isequal("<unk>"), tokens) )

    m = SkipGramModel(size(E,2), size(E,1))
    m.emb.weight .= E                 # copy into freshly inited model
    return m, vocab
end




end # module
