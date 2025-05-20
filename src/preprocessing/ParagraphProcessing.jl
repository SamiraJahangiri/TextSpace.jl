
const NL      = r"\R"            # Unicode newline class
const PAR_SEP = r"\R{2,}"        # ≥ two consecutive newlines

unwrap_lines(txt::AbstractString) =
    replace(txt, r"([^\.\?!])\R(?!\R)" => s"\1 ")   # newline w/o sentence end → space


"""
    split_paragraphs(text; unwrap=false, normalize=true) → Vector{String}

Splits on **blank lines** (1+ empty lines).  
If `unwrap=true`, collapses hard-wrapped lines first.
"""
function split_paragraphs(text::AbstractString;
    unwrap::Bool = true,
    normalize::Bool = true)

    #coarse split on **blank lines**
    paras = split(text, PAR_SEP; keepempty = false)

    #optional hard-wrap removal *within* each paragraph
    unwrap   && (paras = [unwrap_lines(p) for p in paras])
    normalize && (paras = [normalize_whitespace(String(strip(p))) for p in paras])

    return paras
end


"""
    filter_paragraphs(paras; min_chars=25) → Vector{String}
"""
filter_paragraphs(paras::Vector{String}; min_chars::Int=25) =
    [p for p in paras if length(p) ≥ min_chars]


"""
    paragraph_windows(paras, max_tokens;
                      stride=max_tokens,
                      tokenizer) → Iterator

Yield successive paragraph groups whose total token
count (according to `tokenizer`) ≤ `max_tokens`.
"""
function paragraph_windows(paras::Vector{String},
                           max_tokens::Int;
                           stride::Int=max_tokens,
                           tokenizer)

    idxs = 1:length(paras)
    function _iter(state)
        i = state
        i > lastindex(idxs) && return nothing
        j, toks = i, 0
        while j ≤ length(idxs) && toks + length(tokenizer(paras[j])) ≤ max_tokens
            toks += length(tokenizer(paras[j]))
            j += 1
        end
        chunk = paras[i:j-1]
        next_state = i + max(1, stride)
        return chunk, next_state
    end
    return _iter, 1
end


"""
    char_span_to_paragraph_idx(spans, paras) → Vector{Int}

Given vector of `(start,stop)` character spans (1-based offsets in the
original string) return paragraph indices.
"""
function char_span_to_paragraph_idx(spans::Vector{Tuple{Int,Int}},
                                    paras::Vector{String})
    cum = cumsum(length.(paras) .+ 1)   # +1 for newline separator
    map(spans) do (s, _)
        searchsortedfirst(cum, s)
    end
end

