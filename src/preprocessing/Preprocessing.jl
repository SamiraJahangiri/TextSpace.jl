module Preprocessing

include(joinpath(@__DIR__, "CleanText.jl"))
include(joinpath(@__DIR__, "Vocabulary.jl"))          #  -> Vocabulary, convert_tokens_to_ids, …
include(joinpath(@__DIR__, "TextNormalization.jl"))   #  -> clean_text / normalize_whitespace
include(joinpath(@__DIR__, "Stemming.jl"))
include(joinpath(@__DIR__, "Lemmatization.jl"))
include(joinpath(@__DIR__, "Tokenization.jl"))        #  -> tokenize
include(joinpath(@__DIR__, "CharProcessing.jl"))      #  -> tokenize_char
include(joinpath(@__DIR__, "SentenceProcessing.jl"))  #  -> split_sentences
include(joinpath(@__DIR__, "ParagraphProcessing.jl")) #  -> split_paragraphs, paragraph_windows
include(joinpath(@__DIR__, "SubwordTokenization.jl")) #  -> train_bpe, load_bpe, encode, …
include(joinpath(@__DIR__, "TextVectorization.jl"))   #  -> pad_sequences, tfidf_matrix, …
include(joinpath(@__DIR__, "DocumentProcessing.jl"))  #  -> process_document, document_batch_iter


# Export only the high-level pipeline functions
export preprocess_for_char_embeddings,
       preprocess_for_sentence_embeddings,
       preprocess_for_paragraph_embeddings,
       preprocess_for_subword_embeddings,
       preprocess_for_document_embeddings,
       preprocess_for_word_embeddings

# --- Pipeline Function Implementations ---


"""
    preprocess_for_char_embeddings(corpus_input::Union{AbstractString, String};
                                         from_file::Bool = true,
                                         vocab::Union{Vocabulary, Nothing}=nothing,
                                         vocab_options::Dict=Dict(),
                                         clean_options::Dict=Dict(),
                                         char_options::Dict=Dict(),
                                         id_options::Dict=Dict())
                                         -> @NamedTuple{char_ids::Vector{Int},
                                                        vocabulary::Vocabulary,
                                                        chars::Vector{String},
                                                        cleaned_text::String}

Prepares text (from string or file path) for character-level embeddings.
Applies minimal default cleaning (Unicode normalization). Tokenizes into characters (graphemes).
Optionally builds a character vocabulary if one is not provided.

Pipeline (Defaults Applied):
1. Reads corpus if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:none, ...)` using `clean_options`.
3. Tokenizes into characters: `tokenize_char(...)` using `char_options`.
4. Builds vocabulary (if `vocab=nothing`): `build_vocabulary(...)` from characters using `vocab_options`.
5. Converts characters to IDs: `chars_to_ids(...)` using `id_options`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input text (as a single string) or the path to a text file.

# Keyword Arguments
- `vocab::Union{Vocabulary, Nothing}`: A pre-built character `Vocabulary`. If `nothing` (default), a vocabulary is built automatically from the input text characters.
- `vocab_options::Dict`: Options passed to `Vocabulary.build_vocabulary` if `vocab=nothing`. Defaults typically include `min_freq=1` and `special_tokens=["<unk>"]`.
- `clean_options::Dict`: Options passed to `CleanText.clean_text`. Defaults focus on minimal cleaning suitable for char models (e.g., `unicode_normalize=true`, `case_transform=:none`).
- `char_options::Dict`: Options passed to `CharProcessing.tokenize_char` (e.g., `Dict(:keep_space=>true)`).
- `id_options::Dict`: Options passed to `CharProcessing.chars_to_ids` (e.g., `Dict(:add_new=>false)`).

# Returns
- `NamedTuple`: A named tuple containing:
    - `char_ids::Vector{Int}`: The flat list of character IDs.
    - `vocabulary::Vocabulary`: The vocabulary used (provided or built).
    - `chars::Vector{String}`: The list of characters (graphemes) after cleaning.
    - `cleaned_text::String`: The cleaned text before character tokenization.

# Examples

```julia
# Example 1: Basic usage, build vocabulary automatically
text = "Hello World! 123"
result = preprocess_for_char_embeddings(text)

# Expected output structure (IDs depend on generated vocab):
# (char_ids = [..., ...], 
#  vocabulary = Vocabulary(...), 
#  chars = ["H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d", "!", " ", "1", "2", "3"], 
#  cleaned_text = "Hello World! 123")
```

```julia
# Example 2: Using a pre-built vocab and custom cleaning (lowercase)
char_map = Dict("<unk>" => 1, "h" => 2, "e" => 3, "l" => 4, "o" => 5, " " => 6, "w" => 7, "r" => 8, "d" => 9)
inv_char_map = Dict(v => k for (k, v) in char_map)
my_vocab = VocabularyModule.Vocabulary(char_map, inv_char_map, Dict{Int,Int}(), 1)
text = "Hello World!"
result = preprocess_for_char_embeddings(text; 
    vocab=my_vocab, 
    clean_options=Dict(:case_transform=>:lower, :remove_punctuation=>true)
)

# Expected output (using the provided vocab, punctuation removed):
# (char_ids = [2, 3, 4, 4, 5, 6, 7, 5, 8, 4, 9], 
#  vocabulary = my_vocab, 
#  chars = ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"], 
#  cleaned_text = "hello world")
```
"""
function preprocess_for_char_embeddings(corpus_input::Union{AbstractString, String};
                                        from_file::Bool = true,
                                        vocab::Union{Vocabulary, Nothing}=nothing,
                                        vocab_options::Dict=Dict(),
                                        clean_options::Dict=Dict(),
                                        char_options::Dict=Dict(),
                                        id_options::Dict=Dict())

    
    # Minimal cleaning suitable for character models
    default_clean_options = Dict(
        :unicode_normalize => true,
        :do_remove_accents => false,
        :do_remove_punctuation => false,
        :do_remove_symbols => false,
        :do_remove_emojis => false,
        :case_transform => :none # Usually keep case for char models
    )
    # Merge user options with defaults
    final_clean_options = merge(default_clean_options, clean_options)

    default_id_options = Dict(:add_new=>false, :update_counts=>true)
    final_id_options = merge(default_id_options, id_options)
    

    #read Corpus (if file path)
    text = from_file && isfile(corpus_input) ?
           read(corpus_input, String) :
           String(corpus_input)

    #cleaning (using defaults + user options)
    cleaned_text = clean_text(text; final_clean_options...)

    #character Tokenization
    chars = tokenize_char(cleaned_text; char_options...)

    #determine/Build Vocabulary
    if vocab === nothing
        opts = merge(Dict(:min_freq=>1, :special_tokens=>["<unk>"]), vocab_options)
        haskey(opts, :special_tokens) || (opts[:special_tokens] = ["<unk>"])
    
        vdict   = build_vocabulary(chars; opts...)
        unk_id  = get(vdict["token_to_index"], "<unk>", 0)
        if unk_id == 0
            push!(vdict["index_to_token"], "<unk>")
            vdict["token_to_index"]["<unk>"] = length(vdict["index_to_token"])
            unk_id = vdict["token_to_index"]["<unk>"]
        end
        counts = Dict{Int,Int}()

        if haskey(vdict, "freq")
            for (tok, cnt) in vdict["freq"]
                id = get(vdict["token_to_index"], tok, 0)   # 0 if token was filtered
                id == 0 && continue                         # skip missing tokens
                counts[id] = cnt
            end
        end

    
        final_vocab = Vocabulary(vdict["token_to_index"],
                                 vdict["index_to_token"],
                                 counts,
                                 unk_id)
    else
        final_vocab = vocab
    end

    #convert characters to IDs
    ids = chars_to_ids(chars, final_vocab; final_id_options...)

    return (char_ids=ids,
            vocabulary=final_vocab,
            chars=chars,
            cleaned_text=cleaned_text)
end



"""
    preprocess_for_subword_embeddings(text::String, bpe_tokenizer; clean_options=Dict(), sentence_options=Dict(), encode_options=Dict()) -> Vector{Vector{Int}}

Prepares text for subword-based embeddings (e.g., BERT).

Pipeline:
1. Cleans text using `CleanText.clean_text`.
2. Splits cleaned text into sentences using `SentenceProcessing.split_sentences`.
3. Encodes each sentence into subword IDs using `SubwordTokenization.encode`.

# Arguments
- `text::String`: The input text.
- `bpe_tokenizer`: A pre-trained BPE tokenizer (e.g., from `BytePairEncoding.jl` or loaded via `SubwordTokenization.load_bpe`).

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text`.
- `sentence_options::Dict`: Options passed directly to `SentenceProcessing.split_sentences`.
- `encode_options::Dict`: Options passed directly to `SubwordTokenization.encode` for each sentence (e.g., `Dict(:add_special_tokens=>true)`).

# Retu    - `Vector{Vector{Int}}`: A vector of vectors of subword IDs (one inner vector per sentence).

# Examples

```julia
# Assume `bpe` is a loaded BytePairEncoding tokenizer
# using BytePairEncoding, TextEncodeBase
# bpe = load_bpe("path/to/bpe/model") # Or however it's loaded

# Example 1: Basic usage with a hypothetical BPE tokenizer
text = "Subword example. Another sentence."
# Assuming `bpe` exists and is configured
# result = preprocess_for_subword_embeddings(text, bpe)

# Expected output structure (IDs depend heavily on the BPE model):
# [[id1, id2, id3], [id4, id5]] 

# Example 2: Keep punctuation during cleaning for BPE
text = "Keep punctuation? Yes!"
# result = preprocess_for_subword_embeddings(text, bpe; 
#     clean_options=Dict(:remove_punctuation=>false, :case_transform=>:none)
# )

# Expected output structure (IDs depend heavily on the BPE model):
# [[id_k, id_punc, id_q], [id_y, id_exc]] 
# Note: Actual usage requires a valid `bpe` object.
```
"""
function preprocess_for_subword_embeddings(text::String, bpe_tokenizer;
                                           clean_options::Dict=Dict(),
                                           sentence_options::Dict=Dict(),
                                           encode_options::Dict=Dict())
    # 1. Cleaning
    cleaned_text = clean_text(text; clean_options...)

    # 2. Sentence Splitting
    sentences = split_sentences(cleaned_text; sentence_options...)

    # 3. Encode each sentence
    sentence_ids = [SubwordTokenization.encode(bpe_tokenizer, sentence; encode_options...) for sentence in sentences]

    return sentence_ids
end

 
function preprocess_for_word_embeddings(
        corpus_input::Union{AbstractString,String};
        from_file::Bool     = true,
        min_count::Integer  = 1,
        vocab_options::Dict = Dict(),
        clean_options::Dict = Dict())   # ← NEW

    #  read corpus 
    text = (from_file && isfile(corpus_input)) ?
           read(corpus_input,String) : String(corpus_input)

    #  cleaning 
    default_clean_opts = Dict(
        :unicode_normalize     => true,
        :do_remove_accents     => false,
        :do_remove_punctuation => true,
        :do_remove_symbols     => false,
        :do_remove_emojis      => true,
        :case_transform        => :lower,
    )
    final_clean = merge(default_clean_opts, clean_options)      # MERGE
    cleaned_text = clean_text(text; final_clean...)

    #  tokenisation 
    sentences = split_sentences(cleaned_text)
    default_tok_opts = Dict(
        :strip_punctuation => true,
        :lower             => true,
        :remove_stopwords  => false,
        :lemmatize         => true,
        :stem              => false,
    )
    tokenised = [tokenize(s; default_tok_opts...) for s in sentences]

    #  vocabulary 
    merge!(vocab_options, Dict(
        :min_freq       => min_count,
        :special_tokens => get(vocab_options, :special_tokens, ["<unk>"]),
    ))
    delete!(vocab_options, :min_count)

    if !("<unk>" in vocab_options[:special_tokens])
        push!(vocab_options[:special_tokens], "<unk>")
    end

    vdict  = build_vocabulary(vcat(tokenised...); vocab_options...)
    unk_id = get(vdict["token_to_index"], "<unk>", 0)

    counts = Dict{Int,Int}()
    if haskey(vdict, "freq")
        for (tok,cnt) in vdict["freq"]
            id = get(vdict["token_to_index"], tok, 0)
            id == 0 && continue
            counts[id] = cnt
        end
    end

    vocab = Vocabulary(vdict["token_to_index"], vdict["index_to_token"], counts, unk_id)

    #  ids 
    sent_ids = [tokens_to_ids(toks, vocab; add_new=false) for toks in tokenised]

    return (word_ids            = sent_ids,
            vocabulary          = vocab,
            cleaned_text        = cleaned_text,
            tokenized_sentences = tokenised)
end




"""
    preprocess_for_sentence_embeddings(corpus_input::Union{AbstractString, String};
                                             build_vocab::Bool=false,
                                             vocab_options::Dict=Dict(),
                                             sentence_options::Dict=Dict())
                                             -> @NamedTuple{sentences::Vector{String},
                                                            tokenized_sentences::Vector{Vector{String}},
                                                            sentence_token_ids::Union{Vector{Vector{Int}}, Nothing},
                                                            vocabulary::Union{Vocabulary, Nothing},
                                                            cleaned_text::String}

Prepares a corpus (from string or file path) for sentence-level embeddings.
Applies common cleaning and tokenization defaults (lowercasing, punctuation removal, lemmatization).
Splits into sentences, tokenizes each, and optionally builds a shared vocabulary.

Pipeline (Defaults Applied):
1. Reads corpus if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:lower, remove_punctuation=true, ...)`
3. Splits into sentences: `split_sentences(...)` using `sentence_options`.
4. Tokenizes each sentence: `tokenize(..., strip_punctuation=true, lower=true, lemmatize=true, ...)`
5. (Optional) Builds vocabulary: `build_vocabulary(...)` from all sentence tokens if `build_vocab=true`.
6. (Optional) Converts tokens to IDs: `tokens_to_ids(...)` for each sentence if `build_vocab=true`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input corpus text (as a single string) or the path to a text file.

# Keyword Arguments
- `build_vocab::Bool`: If `true`, builds a vocabulary from all sentence tokens and returns token IDs. Defaults to `false`.
- `vocab_options::Dict`: Options passed to `Vocabulary.build_vocabulary` if `build_vocab=true`.
- `sentence_options::Dict`: Options passed to `SentenceProcessing.split_sentences`.

# Returns
- `NamedTuple`: A named tuple containing:
    - `sentences::Vector{String}`: List of cleaned sentences before tokenization.
    - `tokenized_sentences::Vector{Vector{String}}`: Each sentence as a list of its tokens.
    - `sentence_token_ids::Union{Vector{Vector{Int}}, Nothing}`: Each sentence as a list of token IDs (or `nothing`).
    - `vocabulary::Union{Vocabulary, Nothing}`: The vocabulary built across all sentences (or `nothing`).
    - `cleaned_text::String`: The initial cleaned text before sentence splitting.


# Examples

```julia
# Example 1: Basic usage, get tokenized sentences and build vocab
text = "First sentence. Second sentence, with lemma."
result = preprocess_for_sentence_embeddings(text; build_vocab=true, vocab_options=Dict(:min_freq=>1))

# Expected output structure (IDs depend on generated vocab):
# (sentences = ["first sentence", "second sentence with lemma"], 
#  tokenized_sentences = [["first", "sentence"], ["second", "sentence", "with", "lemma"]], 
#  sentence_token_ids = [[id1, id2], [id3, id2, id4, id5]], 
#  vocabulary = Vocabulary(...), 
#  cleaned_text = "first sentence second sentence with lemma")
```

```julia
# Example 2: Get only cleaned sentences (no tokenization/vocab)
text = "Keep punctuation? Maybe. Yes!"
result = preprocess_for_sentence_embeddings(text; 
   build_vocab=false, 
   clean_options=Dict(:remove_punctuation=>false, :case_transform=>:none) # Keep punctuation and case
)

# Expected output:
# (sentences = ["Keep punctuation?", "Maybe.", "Yes!"], 
#  tokenized_sentences = [["keep", "punctuation"], ["maybe"], ["yes"]], # Tokenization still applies defaults
#  sentence_token_ids = nothing, 
#  vocabulary = nothing, 
#  cleaned_text = "Keep punctuation? Maybe. Yes!") 
# Note: tokenized_sentences still uses default tokenization (lowercase, remove punct, lemmatize)
# If you need tokenized sentences matching the cleaned sentences, you'd need more options or separate steps.
```
"""
function preprocess_for_sentence_embeddings(corpus_input::Union{AbstractString, String};
                                            build_vocab::Bool=false,
                                            vocab_options::Dict=Dict(),
                                            sentence_options::Dict=Dict())

    default_clean_options = Dict(
        :unicode_normalize => true,
        :remove_accents => false,
        :remove_punctuation => true,
                                      
        :remove_symbols => false,
        :remove_emojis => true,
        :case_transform => :lower
    )
    default_tokenize_options = Dict(
        :strip_punctuation => true,
        :lower => true,
        :remove_stopwords => false,
        :lemmatize => true,
        :stem => false
    )
    default_id_options = Dict(:add_new=>false, :update_counts=>true)

    #  Read Corpus (if file path)
    text = if isfile(corpus_input)
        read(corpus_input, String)
    else
        String(corpus_input) # Ensure it's a String
    end

    # Cleaning (using defaults)
    cleaned_text = clean_text(text; default_clean_options...)

    #  Sentence Splitting
    sentences_split = split_sentences(cleaned_text; sentence_options...)

    # Tokenize each sentence (using defaults)
    tokenized_sentences = [Tokenization.tokenize(sent; default_tokenize_options...) for sent in sentences_split]

    # Optional Vocabulary Building and ID Conversion
    sent_vocab = nothing
    sent_token_ids = nothing

    if build_vocab
        # Build vocab from *all* tokens across sentences
        all_tokens = vcat(tokenized_sentences...)
        
        # Ensure <unk> is present for vocabulary building
        vocab_build_opts = merge(Dict(:special_tokens => ["<unk>"]), vocab_options)
        if !("<unk>" in get(vocab_build_opts, :special_tokens, []))
            push!(vocab_build_opts[:special_tokens], "<unk>")
        end

        vocab_dict = build_vocabulary(all_tokens; vocab_build_opts...)

        # Create Vocabulary struct instance
        unk_token = "<unk>"
        unk_id = get(vocab_dict["token_to_index"], unk_token, 0)
        if unk_id == 0
            @warn "\\'\'<unk>\\' token not found in generated vocabulary, using ID 0. Ensure \\'\'<unk>\\' is in special_tokens."
        end
        sent_vocab = VocabularyModule.Vocabulary(vocab_dict["token_to_index"],
                                             vocab_dict["index_to_token"],
                                             Dict{Int,Int}(), # Initialize counts
                                             unk_id)

        # Convert tokens to IDs for each sentence using the built vocab
        sent_token_ids = [Tokenization.tokens_to_ids(tokens, sent_vocab; default_id_options...) for tokens in tokenized_sentences]
    end

    return (sentences=sentences_split, # Return the split sentences
            tokenized_sentences=tokenized_sentences,
            sentence_token_ids=sent_token_ids,
            vocabulary=sent_vocab,
            cleaned_text=cleaned_text)
end





"""
    preprocess_for_paragraph_embeddings(corpus_input::Union{AbstractString, String};
                                              build_vocab::Bool=false,
                                              vocab_options::Dict=Dict(),
                                              paragraph_options::Dict=Dict(),
                                              filter_options::Union{Dict, Nothing}=nothing)
                                              -> @NamedTuple{paragraphs::Vector{String},
                                                             tokenized_paragraphs::Vector{Vector{String}},
                                                             paragraph_token_ids::Union{Vector{Vector{Int}}, Nothing},
                                                             vocabulary::Union{Vocabulary, Nothing},
                                                             cleaned_text::String}

Prepares a corpus (from string or file path) for paragraph-level embeddings.
Applies common cleaning and tokenization defaults (lowercasing, punctuation removal, lemmatization).
Splits into paragraphs, optionally filters them, tokenizes each, and optionally builds a shared vocabulary.

Pipeline (Defaults Applied):
1. Reads corpus if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:lower, remove_punctuation=true, ...)`
3. Splits into paragraphs: `split_paragraphs(...)` using `paragraph_options`.
4. (Optional) Filters paragraphs: `filter_paragraphs(...)` using `filter_options`.
5. Tokenizes each paragraph: `tokenize(..., strip_punctuation=true, lower=true, lemmatize=true, ...)`
6. (Optional) Builds vocabulary: `build_vocabulary(...)` from all paragraph tokens if `build_vocab=true`.
7. (Optional) Converts tokens to IDs: `tokens_to_ids(...)` for each paragraph if `build_vocab=true`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input corpus text (as a single string) or the path to a text file.

# Keyword Arguments
- `build_vocab::Bool`: If `true`, builds a vocabulary from all paragraph tokens and returns token IDs. Defaults to `false`.
- `vocab_options::Dict`: Options passed to `Vocabulary.build_vocabulary` if `build_vocab=true`.
- `paragraph_options::Dict`: Options passed to `ParagraphProcessing.split_paragraphs`.
- `filter_options::Union{Dict, Nothing}`: Options passed to `ParagraphProcessing.filter_paragraphs`. Set to `nothing` (default) to skip filtering.

# Returns
- `NamedTuple`: A named tuple containing:
    - `paragraphs::Vector{String}`: List of cleaned (and potentially filtered) paragraphs before tokenization.
    - `tokenized_paragraphs::Vector{Vector{String}}`: Each paragraph as a list of its tokens.
    - `paragraph_token_ids::Union{Vector{Vector{Int}}, Nothing}`: Each paragraph as a list of token IDs (or `nothing`).
    - `vocabulary::Union{Vocabulary, Nothing}`: The vocabulary built across all paragraphs (or `nothing`).
    - `cleaned_text::String`: The initial cleaned text before paragraph splitting.


# Examples

```julia
# Example 1: Basic usage, build vocabulary
text = "First paragraph.\n\nSecond paragraph, with more words."
result = preprocess_for_paragraph_embeddings(text; build_vocab=true, vocab_options=Dict(:min_freq=>1))

# Expected output structure (IDs depend on generated vocab):
# (paragraphs = ["first paragraph", "second paragraph with more word"], 
#  tokenized_paragraphs = [["first", "paragraph"], ["second", "paragraph", "with", "more", "word"]], 
#  paragraph_token_ids = [[id1, id2], [id3, id2, id4, id5, id6]], 
#  vocabulary = Vocabulary(...), 
#  cleaned_text = "first paragraph second paragraph with more word")
```

```julia
# Example 2: Filter short paragraphs, don't build vocab
text = "Short para.\n\nThis is a much longer paragraph that should pass the filter."
result = preprocess_for_paragraph_embeddings(text; 
   build_vocab=false, 
   filter_options=Dict(:min_chars=>20)
)

# Expected output:
# (paragraphs = ["this be a much long paragraph that should pass the filter"], 
#  tokenized_paragraphs = [["this", "be", "a", "much", "long", "paragraph", "that", "should", "pass", "the", "filter"]], 
#  paragraph_token_ids = nothing, 
#  vocabulary = nothing, 
#  cleaned_text = "short para this be a much long paragraph that should pass the filter")
```
"""
function preprocess_for_paragraph_embeddings(corpus_input::Union{AbstractString, String};
                                             build_vocab::Bool=false,
                                             vocab_options::Dict=Dict(),
                                             paragraph_options::Dict=Dict(),
                                             filter_options::Union{Dict, Nothing}=nothing)

    # --- Define Default Options --- 
    default_clean_options = Dict(
        :unicode_normalize => true,
        :remove_accents => false,
        :remove_punctuation => true,
        :remove_symbols => false,
        :remove_emojis => true,
        :case_transform => :lower
    )
    default_tokenize_options = Dict(
        :strip_punctuation => true,
        :lower => true,
        :remove_stopwords => false,
        :lemmatize => true,
        :stem => false
    )
    default_id_options = Dict(:add_new=>false, :update_counts=>true)

    # Read Corpus (if file path)
    text = if isfile(corpus_input)
        read(corpus_input, String)
    else
        String(corpus_input) # Ensure it's a String
    end

    # Cleaning (using defaults)
    cleaned_text = clean_text(text; default_clean_options...)

    # Paragraph Splitting
    paragraphs_raw = split_paragraphs(cleaned_text; paragraph_options...)

    # Optional Filtering
    paragraphs_filtered = if filter_options !== nothing
        filter_paragraphs(paragraphs_raw; filter_options...)
    else
        paragraphs_raw
    end

    # Tokenize each paragraph (using defaults)
    tokenized_paragraphs = [Tokenization.tokenize(para; default_tokenize_options...) for para in paragraphs_filtered]

    # Optional Vocabulary Building and ID Conversion
    para_vocab = nothing
    para_token_ids = nothing

    if build_vocab
        # Build vocab from *all* tokens across paragraphs
        all_tokens = vcat(tokenized_paragraphs...)
        
        # Ensure <unk> is present for vocabulary building
        vocab_build_opts = merge(Dict(:special_tokens => ["<unk>"]), vocab_options)
        if !("<unk>" in get(vocab_build_opts, :special_tokens, []))
            push!(vocab_build_opts[:special_tokens], "<unk>")
        end

        vocab_dict = build_vocabulary(all_tokens; vocab_build_opts...)

        # Create Vocabulary struct instance
        unk_token = "<unk>"
        unk_id = get(vocab_dict["token_to_index"], unk_token, 0)
        if unk_id == 0
            @warn "\'<unk>\' token not found in generated vocabulary, using ID 0. Ensure \'<unk>\' is in special_tokens."
        end
        para_vocab = VocabularyModule.Vocabulary(vocab_dict["token_to_index"],
                                             vocab_dict["index_to_token"],
                                             Dict{Int,Int}(), # Initialize counts
                                             unk_id)

        # Convert tokens to IDs for each paragraph using the built vocab
        para_token_ids = [Tokenization.tokens_to_ids(tokens, para_vocab; default_id_options...) for tokens in tokenized_paragraphs]
    end

    return (paragraphs=paragraphs_filtered, # Return the filtered paragraphs
            tokenized_paragraphs=tokenized_paragraphs,
            paragraph_token_ids=para_token_ids,
            vocabulary=para_vocab,
            cleaned_text=cleaned_text)
end




"""
    preprocess_for_document_embeddings(corpus_input::Union{AbstractString, String};
                                       build_vocab::Bool=false,
                                       vocab_options::Dict=Dict())
                                       -> @NamedTuple{tokens::Vector{String},
                                                      cleaned_text::String,
                                                      token_ids::Union{Vector{Int}, Nothing},
                                                      vocabulary::Union{Vocabulary, Nothing}}

Prepares a document (from string or file path) for document-level embeddings.
Applies common cleaning and tokenization defaults (lowercasing, punctuation removal, lemmatization).
Optionally builds a vocabulary and returns token IDs.

Pipeline (Defaults Applied):
1. Reads document if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:lower, remove_punctuation=true, remove_symbols=false, remove_emojis=true)`
3. Tokenizes the entire cleaned text: `tokenize(..., strip_punctuation=true, lower=true, lemmatize=true, remove_stopwords=false, stem=false)`
4. (Optional) Builds vocabulary: `build_vocabulary(...)` using `vocab_options` if `build_vocab=true`.
5. (Optional) Converts tokens to IDs: `tokens_to_ids(...)` if `build_vocab=true`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input document text (as a single string) or the path to a text file containing the document.

# Keyword Arguments
- `build_vocab::Bool`: If `true`, builds a vocabulary from the document tokens and returns token IDs. Defaults to `false`.
- `vocab_options::Dict`: Options passed directly to `Vocabulary.build_vocabulary` if `build_vocab=true` (e.g., `Dict(:min_freq=>1, :special_tokens=>["<unk>"])`). Ensure `<unk>` is included.

# Returns
- `NamedTuple`: A named tuple containing:
    - `tokens::Vector{String}`: The flat list of tokens after cleaning and tokenization.
    - `cleaned_text::String`: The document text after cleaning.
    - `token_ids::Union{Vector{Int}, Nothing}`: The flat list of token IDs (or `nothing`).
    - `vocabulary::Union{Vocabulary, Nothing}`: The built `Vocabulary` object (or `nothing`).

# Examples

```julia
# Example 1: Basic usage, get tokens, don't build vocab
text = "This is the entire document. It has two sentences."
result = preprocess_for_document_embeddings(text)

# Expected output:
# (tokens = ["this", "be", "the", "entire", "document", "it", "have", "two", "sentence"], 
#  cleaned_text = "this be the entire document it have two sentence", 
#  token_ids = nothing, 
#  vocabulary = nothing)
```

```julia
# Example 2: Build vocabulary and get token IDs
text = "Another document. It is short."
result = preprocess_for_document_embeddings(text; 
   build_vocab=true, 
   vocab_options=Dict(:min_freq=>1)
)

# Expected output structure (IDs depend on generated vocab):
# (tokens = ["another", "document", "it", "be", "short"], 
#  cleaned_text = "another document it be short", 
#  token_ids = [id1, id2, id3, id4, id5], 
#  vocabulary = Vocabulary(...))
```
"""
function preprocess_for_document_embeddings(corpus_input::Union{AbstractString, String};
                                            build_vocab::Bool=false,
                                            vocab_options::Dict=Dict())

    default_clean_options = Dict(
        :unicode_normalize => true,
        :remove_accents => false,
        :remove_punctuation => true,
        :remove_symbols => false,
        :remove_emojis => true,
        :case_transform => :lower
    )
    # Tokenize the whole document as one sequence
    default_tokenize_options = Dict(
        :strip_punctuation => true,
        :lower => true,
        :remove_stopwords => false,
        :lemmatize => true,
        :stem => false
    )
    default_id_options = Dict(:add_new=>false, :update_counts=>true)

    # Read Corpus (if file path)
    text = if isfile(corpus_input)
        read(corpus_input, String)
    else
        String(corpus_input) # Ensure it's a String
    end

    # Cleaning (using defaults)
    cleaned_text = clean_text(text; default_clean_options...)

    # Tokenize the entire cleaned text (using defaults)
    doc_tokens = Tokenization.tokenize(cleaned_text; default_tokenize_options...)

    # Optional Vocabulary Building and ID Conversion
    doc_vocab = nothing
    doc_token_ids = nothing

    if build_vocab
        # Ensure <unk> is present for vocabulary building
        vocab_build_opts = merge(Dict(:special_tokens => ["<unk>"]), vocab_options)
        if !("<unk>" in get(vocab_build_opts, :special_tokens, []))
            push!(vocab_build_opts[:special_tokens], "<unk>")
        end

        vocab_dict = build_vocabulary(doc_tokens; vocab_build_opts...)

        # Create Vocabulary struct instance
        unk_token = "<unk>"
        unk_id = get(vocab_dict["token_to_index"], unk_token, 0)
        if unk_id == 0
            @warn "\'<unk>\' token not found in generated vocabulary, using ID 0. Ensure \'<unk>\' is in special_tokens."
        end
        doc_vocab = VocabularyModule.Vocabulary(vocab_dict["token_to_index"],
                                            vocab_dict["index_to_token"],
                                            Dict{Int,Int}(), # Initialize counts
                                            unk_id)

        # Convert tokens to IDs using the built vocab
        doc_token_ids = Tokenization.tokens_to_ids(doc_tokens, doc_vocab; default_id_options...)
    end

    return (tokens=doc_tokens,
            cleaned_text=cleaned_text,
            token_ids=doc_token_ids,
            vocabulary=doc_vocab)
end



end #END MODULE

