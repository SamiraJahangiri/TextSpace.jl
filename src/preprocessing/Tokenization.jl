
using Unicode                         # stdlib
import Base: isempty


const STOPWORDS_EN = Set([
    "a","an","and","are","as","at","be","by","for","from","has","he","in",
    "is","it","its","of","on","that","the","to","was","were","will","with"
])

"""load_stopwords(lang="en") → Set{String}"""
load_stopwords(lang::AbstractString = "en") =
    lang == "en" ? STOPWORDS_EN :
    error("No stop-word list bundled for language '$lang'.")


# Regex captures:
#   • words  = 1st branch  [\p{L}\p{N}_']+
#   • punct. = 2nd branch  [^\p{L}\p{N}\s]   (any non-alnum, non-ws)
const TOKEN_REGEX               = r"[\p{L}\p{N}_']+|[^\p{L}\p{N}\s]"
const TOKEN_REGEX_WITH_WS       = r"\s+|[\p{L}\p{N}_']+|[^\p{L}\p{N}\s]"
const WHITESPACE_REGEX          = r"^\s+$"

"""
    basic_tokenize(text; keep_whitespace=false) → Vector{String}

Splits `text` into words and punctuation.  Whitespace is discarded unless
`keep_whitespace=true`.
"""
function basic_tokenize(text::AbstractString; keep_whitespace::Bool=false)
    pat = keep_whitespace ? TOKEN_REGEX_WITH_WS : TOKEN_REGEX
    toks = [m.match for m in eachmatch(pat, text)]
    keep_whitespace ? toks :
        filter(t -> !occursin(WHITESPACE_REGEX, t), toks)
end

# small helper functions
apply_case(tok::AbstractString, lower::Bool) =
    lower ? lowercase(String(tok)) : String(tok)


const _PUNCT_CHARS = Set{Char}(".,!?:;\"'’“”()[]{}<>")  # ← one-time cost
"""strip_punct(tok)  — remove leading/trailing punctuation."""
strip_punct(tok::AbstractString) = strip(String(tok), _PUNCT_CHARS)



ngrams(tokens::Vector{<:AbstractString}, n::Int) =
     n == 1 ? tokens :
     [join(tokens[i:i+n-1], '_') for i in 1:length(tokens)-n+1]



"""
    tokenize(text;
             lang                 = "en",
             strip_punctuation    = true,
             lower                = true,
             remove_stopwords     = false,
             stopwords            = STOPWORDS_EN,
             lemmatize            = false,
             stem                 = false,
             ngram                = 1,
             keep_whitespace      = false)

Returns `Vector{String}` of processed tokens.

*Set `lemmatize=true` or `stem=true` **only** if your `Lemmatization.jl`
or `Stemming.jl` files define `lemmatize(::String, ::String)` /
`stem_token(::String, ::String)`.  Otherwise the flags are ignored.*
"""
function tokenize(text::AbstractString;
                  lang::AbstractString = "en",
                  strip_punctuation::Bool = true,
                  lower::Bool = true,
                  remove_stopwords::Bool = false,
                  stopwords::Set{String} = STOPWORDS_EN,
                  lemmatize::Bool = false,
                  stem::Bool = false,
                  ngram::Int = 1,
                  keep_whitespace::Bool = false)

    toks = basic_tokenize(text; keep_whitespace)

    strip_punctuation && (toks = [strip_punct(t) for t in toks])
    lower             && (toks = [apply_case(t, true) for t in toks])
    remove_stopwords  && (toks = [t for t in toks if !(t in stopwords)])

    # Optional lemmatisation / stemming (only if helpers exist)
    lemmatize && @static if isdefined(Main, :lemmatize)
        toks = [lemmatize(t, lang) for t in toks]
    end
    stem && @static if isdefined(Main, :stem_token)
        toks = [stem_token(t, lang) for t in toks]
    end

    toks = filter(!isempty, toks)               # prune empties
    ngram > 1 && (toks = ngrams(toks, ngram))
    return toks
end


"""tokenize_batch(docs; kwargs...) -> Vector{Vector{String}}"""
tokenize_batch(docs::Vector{<:AbstractString}; kwargs...) =
    [tokenize(d; kwargs...) for d in docs]



"""
    tokens_to_ids(tokens, vocab; add_new=false) -> Vector{Int}
"""
function tokens_to_ids(tokens::Vector{<:AbstractString},
                       vocab::Vocabulary;
                       add_new::Bool = false)

    out = Vector{Int}(undef, length(tokens))
    for (i, tok) in enumerate(tokens)
        id = get(vocab.token2id, tok, vocab.unk_id)
        if id == vocab.unk_id && add_new
            id = length(vocab.id2token) + 1
            vocab.token2id[tok] = id
            push!(vocab.id2token, tok)
        end
        out[i] = id
    end
    return out
end


"""
    docs_to_matrix(token_seqs, vocab; pad_value=vocab.unk_id)

Runs `tokens_to_ids` on each document and pads to a matrix
(using `pad_sequences` from `TextVectorization.jl`).
"""
function docs_to_matrix(token_seqs::Vector{Vector{String}},
                        vocab::Vocabulary;
                        pad_value::Int = vocab.unk_id)

    id_seqs = [tokens_to_ids(ts, vocab) for ts in token_seqs]
    return pad_sequences(id_seqs; pad_value)
end

