

using JSON


struct Vocabulary
    token2id::Dict{String,Int}
    id2token::Vector{String}
    counts::Dict{Int,Int}
    unk_id::Int
end

Vocabulary() = Vocabulary(Dict{String,Int}(), String[], Dict{Int,Int}(), 0)



function build_vocabulary(
    tokens::Vector{String};
    min_freq::Int=0,
    max_vocab_size::Int=typemax(Int),
    special_tokens::Vector{String} = String[]
)
    #count frequencies
    freq = Dict{String, Int}()
    for token in tokens
        freq[token] = get(freq, token, 0) + 1
    end

    #filter by min_freq
    filtered_tokens = Dict{String, Int}()
    for (token, count) in freq
        if count >= min_freq
            filtered_tokens[token] = count
        end
    end

    #sort tokens by frequency (descending)
    sorted_tokens = sort(collect(keys(filtered_tokens)), 
                         by = t -> filtered_tokens[t], 
                         rev = true)

    #truncate if necessary
    if length(sorted_tokens) > max_vocab_size
        sorted_tokens = sorted_tokens[1:max_vocab_size]
    end

    #create token-to-index mapping, with special tokens first
    token_to_index = Dict{String, Int}()
    index_to_token = String[]

    
    # insert special tokens first, but only once each
    for st in special_tokens
        if !haskey(token_to_index, st)          # ← new guard
            push!(index_to_token, st)
            token_to_index[st] = length(index_to_token)
        end
    end

    #insert remaining tokens, skipping duplicates
    for tk in sorted_tokens
        if !haskey(token_to_index, tk)
            push!(index_to_token, tk)
            token_to_index[tk] = length(index_to_token)
        end
    end

    return Dict(
        "token_to_index" => token_to_index,
        "index_to_token" => index_to_token,
        "freq"           => freq
    )
end



function build_vocabulary_bpe(
    corpus::Vector{String};
    vocab_size::Int=30000,
    special_tokens::Vector{String}=String[]
)
    #convert each sentence into a list of "words", each word is "chars plus a special end symbol" or keep it simpler: each word => char-level tokens
    tokenized_sentences = [split(s) for s in corpus]

    #turn each word into a list of characters e.g. "hello" -> ['h','e','l','l','o']
    #store them as joined with spaces, so "hello" -> "h e l l o"
    processed_sentences = Vector{Vector{Vector{String}}}(undef, length(tokenized_sentences))
    for i in eachindex(tokenized_sentences)
        words = tokenized_sentences[i]
        char_tokens = [collect_chars_to_symbols(w) for w in words]
        processed_sentences[i] = char_tokens
    end

    #count frequencies of all symbol pairs repeatedly find the most frequent pair, merge it into a single symbol, update.
    merges = String[]  #store the merges
    current_symbols = extract_all_symbols(processed_sentences)

    #final number of merges + initial single chars to be ~ vocab_size - length(special_tokens)
    while length(merges) < vocab_size
        #count pair frequencies
        pair_freq = count_pair_frequencies(processed_sentences)

        if isempty(pair_freq)
            #no more pairs to merge
            break
        end

        #find the best (most frequent) pair
        best_pair, best_count = findmax(pair_freq)  # (pair, freq)
        if best_count < 2
            #merging any further won't help
            break
        end

        #merge the pair in all sentences
        merge_pair_in_sentences!(processed_sentences, best_pair)
        push!(merges, best_pair)
    end

    #build final subword vocabulary from merges + special tokens
    #every distinct symbol in `processed_sentences` is a subword
    subword_freq = Dict{String, Int}()

    #flatten all sentences -> words -> subwords
    for sentence in processed_sentences
        for word in sentence
            #word is now something like ["h", "e", "ll", "o"] after merges
            for token in word
                subword_freq[token] = get(subword_freq, token, 0) + 1
            end
        end
    end

    #sort subwords by frequency
    subwords_sorted = sort(collect(keys(subword_freq)), 
                           by = s -> subword_freq[s],
                           rev = true)

    #truncate if too large (we used merges, so might not be huge)
    if length(subwords_sorted) > vocab_size
        subwords_sorted = subwords_sorted[1:vocab_size]
    end

    token_to_index = Dict{String, Int}()
    index_to_token = String[]

    #add special tokens first
    for st in special_tokens
        push!(index_to_token, st)
        token_to_index[st] = length(index_to_token)
    end

    #add subwords
    for sb in subwords_sorted
        if !haskey(token_to_index, sb)
            push!(index_to_token, sb)
            token_to_index[sb] = length(index_to_token)
        end
    end

    return Dict(
        "token_to_index" => token_to_index,
        "index_to_token" => index_to_token,
        "freq"           => subword_freq,
        "merges"         => merges  #optional: might be helpful for debugging
    )
end

function collect_chars_to_symbols(word::String)
    return [c for c in word]
end

function extract_all_symbols(processed_sentences::Vector{Vector{Vector{String}}})
    syms = Set{String}()
    for sentence in processed_sentences
        for word in sentence
            for token in word
                push!(syms, token)
            end
        end
    end
    return syms
end


function count_pair_frequencies(processed_sentences::Vector{Vector{Vector{String}}})
    pair_freq = Dict{Tuple{String,String}, Int}()
    for sentence in processed_sentences
        for word in sentence
            for i in 1:(length(word)-1)
                pair = (word[i], word[i+1])
                pair_freq[pair] = get(pair_freq, pair, 0) + 1
            end
        end
    end
    return pair_freq
end


function merge_pair_in_sentences!(
    processed_sentences::Vector{Vector{Vector{String}}},
    best_pair::Tuple{String,String}
)
    (a, b) = best_pair
    new_symbol = a*b  #if a="l", b="l", merged is "ll" 
    for sentence in processed_sentences
        for w in 1:length(sentence)
            word = sentence[w]
            new_word = String[]
            i = 1
            while i <= length(word)
                if i < length(word) && word[i] == a && word[i+1] == b
                    push!(new_word, new_symbol)
                    i += 2
                else
                    push!(new_word, word[i])
                    i += 1
                end
            end
            sentence[w] = new_word
        end
    end
end




function build_vocabulary_wordpiece(
    corpus::Vector{String};
    vocab_size::Int=30000,
    special_tokens::Vector{String}=String[]
)
    #start with a minimal set of tokens: the special tokens + [UNK], if not present
    base_tokens = union(special_tokens, ["[UNK]"])

    token_to_index = Dict{String, Int}()
    index_to_token = String[]
    for t in base_tokens
        push!(index_to_token, t)
        token_to_index[t] = length(index_to_token)
    end

    freq = Dict{String, Int}()

    #do multiple passes to discover new subwords until reaching a vocab_size
    for _pass in 1:5  #arbitrary number of passes
        for sentence in corpus
            words = split(sentence)
            for w in words
                #tokenize w using existing subwords
                i = 1
                while i <= length(w)
                    #try to find the largest subword in the vocabulary that matches w starting at position i
                    sub = ""
                    found = false
                    for j in reverse(i:length(w))
                        candidate = w[i:j]
                        if haskey(token_to_index, candidate)
                            sub = candidate
                            found = true
                            #record freq
                            freq[candidate] = get(freq, candidate, 0) + 1
                            i = j+1
                            break
                        end
                    end
                    if !found
                        #no known subword matched
                        #add a single character or treat it as [UNK]
                        
                        char_sub = w[i]
                        #create a new subword if we have capacity.
                        if length(token_to_index) < vocab_size && !haskey(token_to_index, char_sub)
                            push!(index_to_token, char_sub)
                            token_to_index[char_sub] = length(index_to_token)
                        end
                        freq[char_sub] = get(freq, char_sub, 0) + 1
                        i += 1
                    end
                end
            end
        end
    end

    #now we have a frequency table of discovered subwords sort them and keep top (vocab_size) minus however many special tokens we had
    sorted_subwords = sort(collect(keys(freq)), by = s -> freq[s], rev = true)
    # We want the final size = vocab_size. We already inserted base_tokens at the beginning
    allowed_capacity = vocab_size - length(base_tokens)
    final_subwords = String[]
    for sb in sorted_subwords
        if ! (sb in base_tokens)
            push!(final_subwords, sb)
        end
    end

    if length(final_subwords) > allowed_capacity
        final_subwords = final_subwords[1:allowed_capacity]
    end

    #rebuild token_to_index in correct order
    token_to_index = Dict{String, Int}()
    index_to_token = String[]

    #base_tokens first (special + [UNK])
    for t in base_tokens
        push!(index_to_token, t)
        token_to_index[t] = length(index_to_token)
    end

    #then discovered subwords
    for sb in final_subwords
        push!(index_to_token, sb)
        token_to_index[sb] = length(index_to_token)
    end

    return Dict(
        "token_to_index" => token_to_index,
        "index_to_token" => index_to_token,
        "freq"           => freq
    )
end


"""
    convert_tokens_to_ids(tokens, vocab;
                          add_new=false,
                          update_counts=true) → Vector{Int}

Map string tokens to integer ids using `vocab::Vocabulary`.

* If `add_new=true` unknown tokens are appended to the vocabulary
  (needed while **training** the vocab).
* If `update_counts=true` increments `vocab.counts[id]` for every hit.
"""
function convert_tokens_to_ids(tokens::Vector{String},
                               vocab::Vocabulary;
                               add_new::Bool      = false,
                               update_counts::Bool = true)

    out = Vector{Int}(undef, length(tokens))
    for (i, tok) in enumerate(tokens)

        id = get(vocab.token2id, tok, 0)        # 0 means OOV for now

        if id == 0
            if add_new
                id = length(vocab.id2token) + 1
                vocab.token2id[tok] = id
                push!(vocab.id2token, tok)
                update_counts && (vocab.counts[id] = 1)
            else
                id = vocab.unk_id
                update_counts && (vocab.counts[id] = get(vocab.counts, id, 0) + 1)
            end
        else
            update_counts && (vocab.counts[id] = get(vocab.counts, id, 0) + 1)
        end

        out[i] = id
    end
    return out
end


"""
    convert_ids_to_tokens(ids, vocab) → Vector{String}

Inverse mapping. Unknown ids return `"<unk>"` by convention.
"""
function convert_ids_to_tokens(ids::Vector{<:Integer}, vocab::Vocabulary)
    unk = "<unk>"
    toks = Vector{String}(undef, length(ids))
    for (i, id) in enumerate(ids)
        toks[i] = 1 ≤ id ≤ length(vocab.id2token) ? vocab.id2token[id] : unk
    end
    return toks
end


"""
    convert_batch_tokens_to_ids(docs, vocab;
                                pad_value = vocab.unk_id,
                                kwargs...) → Matrix{Int}

High-level helper: calls `convert_tokens_to_ids` on each document and
pads to a column-major matrix via `pad_sequences` (from TextVectorization.jl).
`docs` is a `Vector{Vector{String}}`.
"""
function convert_batch_tokens_to_ids(docs::Vector{Vector{String}},
                                     vocab::Vocabulary;
                                     pad_value::Int = vocab.unk_id,
                                     kwargs...)

    seqs = [convert_tokens_to_ids(d, vocab; kwargs...) for d in docs]
    pad_sequences(seqs; pad_value)
end


function save_vocabulary(vocab::Dict, filename::String)
    open(filename, "w") do io
        JSON.print(io, vocab)
    end
end

function load_vocabulary(filename::String)
    return JSON.parsefile(filename)
end