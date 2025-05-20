
const DEFAULTS = Dict(
    :unwrap_lines        => true,
    :paragraph_min_chars => 25,
    :sentence_split      => true,
    :lower               => true,
    :strip_punctuation   => true,
    :remove_stopwords    => false,
    :lemmatize           => false,
    :stem                => false,
    :ngram               => 1,
    :add_special_tokens  => true,
    :pad_value           => 0
)


"""
    process_document(text;
                     tok,                   # required SubwordTokenizer
                     vocab = nothing,       # optional Vocabulary (word-level)
                     return = :batch,       # :batch | :ids | :tokens | :sentences
                     kwargs...)

End-to-end preprocessing.

*Paragraph → Sentence splitting* happens automatically if `sentence_split=true`.  
The BPE tokenizer (`tok`) is mandatory when you request `:ids` or `:batch`.

`return` options  
| value        | output type                              |
|--------------|------------------------------------------|
| `:batch`     | padded matrix `(max_len, num_units)`     |
| `:ids`       | `Vector{Vector{Int}}`                    |
| `:tokens`    | `Vector{Vector{String}}`                 |
| `:sentences` | `Vector{String}` (after splitting)       |
"""
function process_document(text::AbstractString,
    tok;                              # SubwordTokenizer
    vocab::Union{Nothing,Vocabulary} = nothing,
    mode::Symbol                     = :batch,
    kwargs...)

    opt = merge(DEFAULTS, Dict(kwargs))

    paras = split_paragraphs(text; unwrap = opt[:unwrap_lines])
    paras = filter_paragraphs(paras; min_chars = opt[:paragraph_min_chars])

    sentences = opt[:sentence_split] ? vcat([split_sentences(p) for p in paras]...) :
    paras

    if mode == :sentences
        return sentences
    end

    token_seqs = tokenize_batch(
        sentences;
        strip_punctuation = opt[:strip_punctuation],
        lower             = opt[:lower],
        remove_stopwords  = opt[:remove_stopwords],
        lemmatize         = opt[:lemmatize],
        stem              = opt[:stem],
        ngram             = opt[:ngram]
    )

    if mode == :tokens
        return token_seqs
    end

    if vocab === nothing                       # sub-word route
        id_seqs = [encode(tok, join(ts, ' '); add_special_tokens = opt[:add_special_tokens])  for ts in token_seqs]
        pad_val = tok.pad_id
    else                                        # word route
        id_seqs = [convert_tokens_to_ids(ts, vocab) for ts in token_seqs]
        pad_val = vocab.unk_id
    end

    if mode == :ids
        return id_seqs
    end

    return pad_sequences(id_seqs; pad_value = pad_val)
end



"""
    document_batch_iter(text;
                        tok,
                        vocab = nothing,
                        max_tokens = 256,
                        kwargs...)

Stream paragraph windows so each batch has ≤ `max_tokens` tokens
(according to the chosen tokenizer).  Useful for long docs.
Returns an iterator of padded matrices.
"""

function document_batch_iter(text, tok;
    vocab       = nothing,
    max_tokens::Int = 256,
    kwargs...)

    paras = split_paragraphs(text; unwrap = true)
    paras = filter_paragraphs(paras; min_chars = 25)

    win_iter, first_state = paragraph_windows(
        paras, max_tokens;
        stride    = max_tokens,
        tokenizer = p -> encode(tok, p)
    )

    function _iter(state)
    data = iterate(win_iter, state)         # (chunk, next_state) or nothing
    data === nothing && return nothing

    chunk, nxt = data
    batch = process_document(join(chunk, "\n\n"), tok;
            vocab = vocab,
            mode  = :batch,
            kwargs...)
    return batch, nxt
    end

    return _iter, first_state
end

