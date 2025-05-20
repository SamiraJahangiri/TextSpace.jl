
using BytePairEncoding
import Serialization

const DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]

"""
    train_bpe(corpus_paths;
              vocab_size           = 32_000,
              special_tokens       = DEFAULT_SPECIAL_TOKENS,
              num_merges           = 1_000_000,
              model_path           = "bpe.model")

Learn a BPE tokenizer from the text files listed in `corpus_paths`
and serialize it to `model_path`.  
Returns the path to the saved model.
"""
function train_bpe(corpus_paths::Vector{String};
                   vocab_size::Int           = 32_000,
                   special_tokens::Vector{String} = DEFAULT_SPECIAL_TOKENS,
                   num_merges::Int           = 1_000_000,
                   model_path::AbstractString = "bpe.model")

    corpus = vcat(readlines.(corpus_paths)...)  # collect all lines
    tokenizer = learn_bpe(
        corpus;
        vocab_size         = vocab_size,
        num_merges         = num_merges,
        special_tokens     = special_tokens,
        ordered_specials   = true,
    )

    Serialization.serialize(model_path, tokenizer)
    return model_path
end


"""
    load_bpe(path) -> tokenizer

Deserialize a previously trained BPE tokenizer.
"""
load_bpe(path::AbstractString) = Serialization.deserialize(path)


"""
    encode(tok, text; add_special_tokens=false) → Vector{Int}

Convert `text` to a vector of token-ids.  
If `add_special_tokens` is true the sequence becomes

<cls> …tokens… <sep>

"""
function encode(tok, text::AbstractString; add_special_tokens::Bool=false)
    ids = encode(tok, text) # BytePairEncoding.encode
    return add_special_tokens ?
           [tok.vocab["<cls>"]; ids; tok.vocab["<sep>"]] :
           ids
end


"""
    encode_batch(tok, docs; pad_id=tok.vocab["<pad>"], add_special_tokens=false)

Vectorised version that returns a **column-major matrix**
`(max_len, batch)`, padded with `pad_id`.
"""
function encode_batch(tok,
                      docs::Vector{<:AbstractString};
                      pad_id::Integer       = tok.vocab["<pad>"],
                      add_special_tokens::Bool = false)

    seqs   = [encode(tok, d; add_special_tokens) for d in docs]
    maxlen = maximum(length.(seqs))
    mat    = fill(pad_id, maxlen, length(seqs))

    for (i, s) in enumerate(seqs)
        mat[1:length(s), i] .= s
    end
    return mat
end


"""
    decode(tok, ids) → String

Inverse of `encode`.
"""
decode(tok, ids::Vector{<:Integer}) = decode(tok, ids)      # BytePairEncoding.decode


"""
    vocabulary(tok) → Dict{String,Int}

Expose the internal token→id map so it can be wrapped by your own
`Vocabulary` struct if desired.
"""
vocabulary(tok) = tok.vocab


"""
    special_id(tok, sym) → Int

Convenience accessor for special token ids, e.g.

```julia
mask = special_id(tok, "<mask>")

"""
special_id(tok, sym::AbstractString) = tok.vocab[sym]