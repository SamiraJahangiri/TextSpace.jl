
using LinearAlgebra
using Statistics


"""
    pad_sequences(seqs; maxlen = nothing, pad_value = 0, trunc = :post)

Return a column-major matrix where each column is a padded/truncated sequence.

* `maxlen`   — target length.  If `nothing` use the longest sequence.
* `pad_value`— value to insert in empty positions.
* `trunc`    — `:pre` or `:post` (keep head or tail when truncating).
"""
function pad_sequences(seqs::Vector{<:Vector{<:Integer}};
    maxlen::Union{Nothing,Int}=nothing,
    pad_value::Integer = 0,
    trunc::Symbol = :post)


    maxlen === nothing && (maxlen = maximum(length.(seqs)))
    m   = fill(pad_value, maxlen, length(seqs))
    for (i, s) in enumerate(seqs)
        if length(s) > maxlen               # truncation
            trunc === :post ? (s = s[1:maxlen]) :
                               (s = s[end-maxlen+1:end])
        end
        m[1:length(s), i] .= s              # copy
    end
    return m
end



"""
    one_hot(seq, vocab_size) → Matrix{Bool}

Column-major binary matrix `(vocab_size, length(seq))`.
"""
function one_hot(seq::Vector{<:Integer}, vocab_size::Integer)
    mat = falses(vocab_size, length(seq))
    for (j, id) in enumerate(seq)
        1 ≤ id ≤ vocab_size && (mat[id, j] = true)
    end
    return mat
end


"""
    bow_counts(seq, vocab_size) → Vector{Int}

Simple term-frequency vector (Bag-of-Words).
"""
function bow_counts(seq::Vector{<:Integer}, vocab_size::Integer)
    v = zeros(Int, vocab_size)
    for id in seq
        1 ≤ id ≤ vocab_size && (v[id] += 1)
    end
    return v
end


"""
    bow_matrix(docs, vocab_size) → Matrix{Int}

Stack `bow_counts` column-wise for many documents.
"""
function bow_matrix(docs::Vector{<:Vector{<:Integer}}, vocab_size::Integer)
    mat = zeros(Int, vocab_size, length(docs))
    for (i, d) in enumerate(docs)
        mat[:, i] .= bow_counts(d, vocab_size)
    end
    return mat
end


"""
    tfidf_matrix(docs, vocab_size; smooth_idf = 1.) → Matrix{Float64}

* `smooth_idf` controls additive smoothing (set to 0 for raw).
"""
function tfidf_matrix(docs::Vector{<:Vector{<:Integer}}, vocab_size::Integer;
                      smooth_idf::Real = 1.0)

    tf = bow_matrix(docs, vocab_size)               # term frequency
    df = vec(sum(tf .> 0; dims=2))                  # doc frequency
    idf = log.((length(docs) .+ smooth_idf) ./ (df .+ smooth_idf))
    return tf .* idf
end



"""
    batch_iter(seqs, batch_size;
               shuffle = true,
               pad_value = 0,
               rng = Random.GLOBAL_RNG)

Return an iterator that yields padded mini-batches (matrix, raw_subset).
"""
function batch_iter(seqs::Vector{<:Vector{Int}},
                    batch_size::Integer;
                    shuffle::Bool = true,
                    pad_value::Int = 0,
                    rng = Random.GLOBAL_RNG)

    idx = collect(eachindex(seqs))
    shuffle && Random.shuffle!(rng, idx)

    function _iter(state)
        start = state
        start > length(idx) && return nothing
        stop  = min(start + batch_size - 1, length(idx))
        subset = idx[start:stop]
        matrix = pad_sequences(seqs[subset]; pad_value)
        new_state = stop + 1
        return (matrix, subset), new_state
    end
    return _iter, 1
end

