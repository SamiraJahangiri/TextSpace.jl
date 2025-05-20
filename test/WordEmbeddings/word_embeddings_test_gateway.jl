
using Test, Random
using TextSpace                  # pulls in CharacterEmbeddings
using Statistics

using TextSpace.WordEmbeddings.WordEmbeddingUtilities: windowify, build_word_pairs

const WE = TextSpace.WordEmbeddings
const V  = TextSpace.Preprocessing.Vocabulary  
const PP   = TextSpace.Preprocessing
const WU = TextSpace.WordEmbeddings.WordEmbeddingUtilities


make_vocab(N) = let
    tok2id = Dict("<unk>" => 1)
    for i in 1:N
        tok2id[string(i)] = i + 1
    end
    id2tok = Vector{String}(undef, length(tok2id))
    for (tok, id) in tok2id
        id2tok[id] = tok
    end
    V(tok2id, id2tok, Dict{Int,Int}(), 1)
end



@testset "Word-Embeddings - training smoke-run" begin
    vocab  = make_vocab(6)                       # ids 2…7 are "words"
    ids    = rand(2:7, 600)                      # fake corpus (no OOV)

    model = WE.train!(ids, vocab;
                      objective = :skipgram,     # or :cbow
                      emb_dim   = 16,
                      batch     = 128,
                      epochs    = 1,
                      k_neg     = 2,
                      lr        = 1e-1,
                      rng       = MersenneTwister(42))

    E = WE.embeddings(model)                     # 16 × |V|
    @test size(E) == (16, length(vocab.id2token))
    @test std(vec(E)) > 0                        # not all identical / zero
    @test all(isfinite, E)

    # round-trip through disk
    tmp = tempname() * ".tsv"
    WE.save_embeddings(tmp, model, vocab)
    @test isfile(tmp)

    model2, vocab2 = WE.load_embeddings(tmp)
    @test E ≈ WE.embeddings(model2)              # identical weights
    @test vocab.token2id == vocab2.token2id      # identical vocab mapping
    rm(tmp; force = true)
end


"helper: build a toy word vocabulary with <unk>=1"
make_vocab(N) = let
    tok2id = Dict("<unk>" => 1)
    for i in 1:N
        tok2id[string(i)] = i + 1
    end
    id2tok = Vector{String}(undef, length(tok2id))
    for (tok, id) in tok2id
        id2tok[id] = tok
    end
    V(tok2id, id2tok, Dict{Int,Int}(), 1)       # 1 is the unk-id
end

@testset "deterministic Skip-Gram" begin
    rng_pairs = MersenneTwister(77)
    ids       = rand(rng_pairs, 2:7, 1_000)    # corpus
    vocab     = make_vocab(6)

    Random.seed!(123)
    m1 = WE.train!(ids, vocab; epochs=1, rng = copy(rng_pairs))

    Random.seed!(123)
    m2 = WE.train!(ids, vocab; epochs=1, rng = copy(rng_pairs))

    @test isequal(WE.embeddings(m1), WE.embeddings(m2))
end

@testset "SG loss drops" begin
    ids   = repeat(2:5, 250)                 # small synthetic corpus
    vocab = make_vocab(4)

    # fixed probe pairs
    C,O   = WE.WordEmbeddingUtilities.build_word_pairs(
                ids; mode=:skipgram, win=32, stride=32, radius=2,
                rng=MersenneTwister(9))
    C, O  = C[1:128], O[1:128]

    init  = WE.SkipGramModel(length(vocab.id2token), 16)
    l0    = WE.sg_loss(init, C, O, repeat(C,5), rand(1:length(vocab.id2token),length(O)*5))

    trained = WE.train!(ids, vocab; epochs=5, emb_dim=16, batch=256)
    l1 = WE.sg_loss(trained, C, O, repeat(C,5), rand(1:length(vocab.id2token),length(O)*5))

    @test l1 < l0
end

@testset "OOV handled as <unk>" begin
    vocab = make_vocab(6)                    #  <unk> has id = 1
    m     = WE.SkipGramModel(length(vocab.id2token), 8)

    # both unknown tokens map to the <unk> column
    unk_vec = WE.vector(m, vocab, "<unk>")
    @test WE.vector(m, vocab, "not-in-vocab") === unk_vec
    @test WE.vector(m, vocab, "💥")           === unk_vec
end


@testset "nearest-neighbour sanity" begin
    using LinearAlgebra: dot, norm            # local import

    vocab = make_vocab(5)
    ids   = repeat(2:6, 200)
    m     = WE.train!(ids, vocab; epochs = 1, emb_dim = 8)

    E   = WE.embeddings(m)
    cos(a, b) = dot(a, b) / (norm(a) * norm(b) + eps())

    sims = [cos(E[:, 2], E[:, j]) for j in 1:size(E, 2)]   # token '1'
    @test argmax(sims) == 2                                # self-similarity max
end


@testset "loss improves with more epochs - WordEmbeddings" begin
    #  synthetic corpus 
    big_ids = repeat(2:6, 2_000) # 10 000 word-ids (skip 1=<unk>)
    vocab   = make_vocab(5)       # |V| = 6,  id 1 = <unk>

    #  fixed probe batch (  same random seed each run) 
    using TextSpace.WordEmbeddings.WordEmbeddingUtilities: build_word_pairs
    Cp, Op = build_word_pairs(big_ids; mode = :skipgram,
                              win = 40, stride = 40,
                              radius = 2, rng = MersenneTwister(90))
    Cp, Op = Cp[1:100], Op[1:100]                 # 100 probe pairs

    kneg   = 5
    neg_O  = rand(MersenneTwister(91),
                  1:length(vocab.id2token), length(Op)*kneg)
    neg_C  = vcat(ntuple(_->Cp, kneg)...)         # cheap repeat

    #  train 3 epochs vs 6 epochs from the *same* init 
    Random.seed!(555)
    m3 = WE.train!(big_ids, vocab;
                   epochs = 3, emb_dim = 32, lr = 0.02,
                   batch  = 256, rng = MersenneTwister(1))

    Random.seed!(555)        # identical initial weights again
    m6 = WE.train!(big_ids, vocab;
                   epochs = 6, emb_dim = 32, lr = 0.02,
                   batch  = 256, rng = MersenneTwister(1))

    l3 = WE.sg_loss(m3, Cp, Op, neg_C, neg_O)
    l6 = WE.sg_loss(m6, Cp, Op, neg_C, neg_O)

    @test l6 <= l3            # more training never worse
end


@testset "loss improves with more epochs - WordEmbeddings" begin
    #  synthetic corpus 
    big_ids = repeat(2:6, 2_000)   # 10 000 word-ids (skip 1=<unk>)
    vocab   = make_vocab(5)       # |V| = 6,  id 1 = <unk>

    #  fixed probe batch (    same random seed each run) ----
    using TextSpace.WordEmbeddings.WordEmbeddingUtilities: build_word_pairs
    Cp, Op = build_word_pairs(big_ids; mode = :skipgram,
                              win = 40, stride = 40,
                              radius = 2, rng = MersenneTwister(90))
    Cp, Op = Cp[1:100], Op[1:100]                 # 100 probe pairs

    kneg   = 5
    neg_O  = rand(MersenneTwister(91),
                  1:length(vocab.id2token), length(Op)*kneg)
    neg_C  = vcat(ntuple(_->Cp, kneg)...)         # cheap repeat

    #  train 3 epochs vs 6 epochs from the *same* init 
    Random.seed!(555)
    m3 = WE.train!(big_ids, vocab;
                   epochs = 3, emb_dim = 32, lr = 0.02,
                   batch  = 256, rng = MersenneTwister(1))

    Random.seed!(555)        # identical initial weights again
    m6 = WE.train!(big_ids, vocab;
                   epochs = 6, emb_dim = 32, lr = 0.02,
                   batch  = 256, rng = MersenneTwister(1))

    l3 = WE.sg_loss(m3, Cp, Op, neg_C, neg_O)
    l6 = WE.sg_loss(m6, Cp, Op, neg_C, neg_O)

    @test l6 <= l3            # more training never worse
end


@testset "save / reload - WordEmbeddings" begin
    #  tiny corpus & vocab 
    ids   = repeat(2:4, 100)                     # 300 tokens, ids 2,3,4
    vocab = make_vocab(3)                        # <unk>=1, words=2:4

    model = WE.train!(ids, vocab; epochs = 1,
                      emb_dim = 8, batch = 64)

    #  save to disk 
    tmp = tempname()*".tsv"
    WE.save_embeddings(tmp, model, vocab)
    @test isfile(tmp)

    #  read back & compare 
    cols = (parse.(Float64, split(l, '\t')[2:end]) for l in eachline(tmp))
    Efile = hcat(cols...) |> Matrix               # emb_dim × |V|

    @test Efile ≈ WE.embeddings(model)            # exact round-trip
    rm(tmp; force = true)
end


@testset "WE - OOV handled as <unk>" begin
    vocab = make_vocab(3)                      # <unk>=1, words=2:4
    m     = WE.SkipGramModel(4, 8)             # emb_dim = 8

    unk = WE.vector(m, vocab, "<unk>")
    @test WE.vector(m, vocab, "totally-new") === unk
end


@testset "WE : batch-boundary safety" begin
    vocab  = make_vocab(5)
    ids    = rand(2:6, 1234)                   # 1234 % 256 ≠ 0
    m      = WE.train!(ids, vocab;
                       epochs = 1, batch = 256, emb_dim = 16)

    @test !any(isnan, WE.embeddings(m))        # no NaNs after training
end

@testset "WE : dot_scores throws on OOV id" begin
    m = WE.SkipGramModel(4, 8)
    @test_throws BoundsError WE.dot_scores(m, [2], [5])   # ctx id 5 invalid
end


@testset "pipeline smoke-run (word)" begin
    # create a toy corpus 
    raw  = repeat("Hello fresh new world! ", 30)      # 4 × 30 = 120 tokens
    toks = split(lowercase(raw))                      

    # build a tiny vocabulary  (<unk> has id 1)
    tok2id = Dict("<unk>" => 1)
    for (i, t) in enumerate(unique(toks))
        tok2id[t] = i + 1
    end

    ids   = [get(tok2id, t, 1) for t in toks]

    # id2tok in index-order (avoids the deprecated Dict-sort)
    id2tok = Vector{String}(undef, length(tok2id))
    for (tok, id) in tok2id
        id2tok[id] = tok
    end
    vocab = V(tok2id, id2tok, Dict{Int,Int}(), 1)

    @test length(ids) > 100          # now passes (120 > 100)
    @test haskey(vocab.token2id, "hello")

    # tiny training run 
    m = WE.train!(ids, vocab;
                  objective = :skipgram,
                  emb_dim   = 16,
                  batch     = 32,
                  epochs    = 1,
                  rng       = MersenneTwister(123))

    E = WE.embeddings(m)
    @test size(E, 2) == length(vocab.id2token)
    @test !any(isnan, E)
end

@testset "negative-sampling sweep (word)" begin
    ids    = repeat(2:5, 400)                   # fake corpus of four words
    vocab  = make_vocab(4)                      # ids 2:5 are words

    for k in 1:6
        m = WE.train!(ids, vocab;
                      objective = :skipgram,
                      epochs    = 1,
                      emb_dim   = 8,
                      batch     = 128,
                      k_neg     = k,
                      rng       = MersenneTwister(100+k))
        @test !any(isnan, WE.embeddings(m))
    end
end

@testset "batch size = 1 edge-case (word)" begin
    ids   = rand(2:6, 150)                      # 150 tokens, 5 words
    vocab = make_vocab(5)

    m = WE.train!(ids, vocab;
                  epochs = 1,
                  emb_dim = 8,
                  batch  = 1,                   # single-sample batches
                  rng    = MersenneTwister(7))

    @test !any(isnan, WE.embeddings(m))
end


@testset "windowify & pair-builder edge-cases" begin
    using TextSpace.WordEmbeddings.WordEmbeddingUtilities

    ids = 1:10 |> collect                    # 10 dummy tokens

    @test windowify(ids; win = 4, stride = 2) |> length == 4
    @test windowify(ids; win = 99, stride = 5) == Vector{SubArray{Int}}()  # oversize win

    # radius larger than window → no pairs, but also no throw
    C,O = build_word_pairs(ids; mode = :cbow, win = 3, stride = 3, radius = 5)
    @test isempty(C) && isempty(O)
end


@testset "OOV handled as <unk> (word)" begin
    vocab = make_vocab(5)              # <unk> is id 1
    m     = WE.SkipGramModel(6, 8)     # |V| = 6 (5 words + <unk>)
    unk   = WE.vector(m, vocab, "<unk>")

    @test WE.vector(m, vocab, "does-not-exist") === unk
end

@testset "deterministic CBOW (word)" begin
    ids   = rand(MersenneTwister(9), 2:7, 1_000)
    vocab = make_vocab(6)

    Random.seed!(123);  m1 = WE.train!(ids, vocab; objective = :cbow,
                                       epochs = 1, emb_dim = 8,
                                       rng = MersenneTwister(99))

    Random.seed!(123);  m2 = WE.train!(ids, vocab; objective = :cbow,
                                       epochs = 1, emb_dim = 8,
                                       rng = MersenneTwister(99))

    @test WE.embeddings(m1) == WE.embeddings(m2)
end


@testset "CBOW loss drops (word)" begin
    ids   = repeat(2:5, 1_250)                 # 5 000 tokens
    vocab = make_vocab(4)

    
    C, O = WE.WordEmbeddingUtilities.build_word_pairs(
              ids; mode = :cbow, win = 20, stride = 20, radius = 1,
              rng  = MersenneTwister(11))
    C, O = C[1:256], O[1:256]                  # 256 positive pairs

    kneg   = 5
    neg_C  = vcat(Iterators.repeated(C, kneg)...)      
    rng123 = MersenneTwister(123)
    neg_O  = rand(rng123, 1:length(vocab.id2token), length(O)*kneg)

    # baseline (random weights)
    m0 = WE.SkipGramModel(length(vocab.id2token), 16)
    l0 = WE.cbow_loss(m0, C, O, neg_C, neg_O)

    # train a CBOW model
    m1 = WE.train!(ids, vocab;
                   objective = :cbow,
                   epochs    = 8,
                   emb_dim   = 16,
                   batch     = 128,
                   lr        = 5e-2,
                   rng       = MersenneTwister(321))

    l1 = WE.cbow_loss(m1, C, O, neg_C, neg_O)

    @test l1 < l0                     # definite improvement
end



@testset "k_neg = 0 & batch = 1 (word)" begin
    ids   = rand(2:6, 200)
    vocab = make_vocab(5)

    m = WE.train!(ids, vocab; epochs = 1, batch = 1,
                  k_neg = 0, emb_dim = 8,
                  rng = MersenneTwister(7))

    @test !any(isnan, WE.embeddings(m))
end


@testset "pipeline smoke-run (word) - small real text" begin
    raw = lowercase("The quick brown fox jumps over the lazy dog." ^ 15)

    prep  = TextSpace.Preprocessing.preprocess_for_word_embeddings(
                raw; min_count = 1, from_file = false)

    ids_flat = vcat(prep.word_ids...)          # flatten sentences
    vocab     = prep.vocabulary

    m = WE.train!(ids_flat, vocab; epochs = 2,
                  emb_dim = 32, batch = 32)

    # cosine self-similarity sanity
    E   = WE.embeddings(m)
    idt = vocab.token2id["the"]
    cos(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())
    sims = [cos(E[:,idt], E[:,j]) for j in 1:size(E,2)]
    @test argmax(sims) == idt                  #'the' is closest to itself
end

 
@testset "Unicode corpus - word embeddings" begin
    # aroud 320 tokens after tokenisation a long enough for the window builder
    txt = repeat("🎉 café λόγος lait mówisz word yo ☕ - vraiment!  ", 40)

    mktemp() do path, io
        write(io, txt); close(io)

        prep = PP.preprocess_for_word_embeddings(path;
                  from_file    = true,
                  min_count    = 1,
                  # KEEP emojis & punctuation so they appear in the vocab
                  clean_options = Dict(
                      :do_remove_emojis      => false,
                      :do_remove_punctuation => false),
                  vocab_options = Dict())   # defaults fine

        ids   = vcat(prep.word_ids...)      # flatten sentences
        vocab = prep.vocabulary

        @test length(ids) > 300             # plenty of word-ids
        @test haskey(vocab.token2id, "🎉")  # emoji survived
        @test haskey(vocab.token2id, "café")

        m = WE.train!(ids, vocab;
                      objective = :cbow,
                      epochs    = 1,
                      emb_dim   = 16,
                      batch     = 64,
                      rng       = MersenneTwister(99))

        @test !any(isnan, WE.embeddings(m))
    end
end


@testset "real-text corpus smoke-run (word)" begin
    # an English excerpt - plenty of distinct words
    excerpt = """
        The Jedi have always traveled the stars, defending peace and justice across the galaxy.
        But the galaxy is changing, and the Jedi Order along with it.
        More and more, the Order finds itself focused on the future of the Republic, secluded on Coruscant,
        where the twelve members of the Jedi Council weigh crises on a galactic scale.
        As yet another Jedi Outpost left over from the Republic's golden age is set to be decommissioned on the
        planet Kwenn, Qui-Gon Jinn challenges the Council about the Order's increasing isolation.
        Mace Windu suggests a bold response: all twelve Jedi Masters will embark on a goodwill
        mission to help the planet and to remind the people of the galaxy that the
        Jedi remain as stalwart and present as they have been across the ages.
        """

    mktemp() do path, io
        write(io, excerpt); close(io)

        prep = TextSpace.Preprocessing.preprocess_for_word_embeddings(
                   path; from_file = true,          # file-path branch
                   min_count = 1,                   # keep every token
                   clean_options = Dict(            # keep punctuation etc.
                       :do_remove_punctuation => false,
                       :do_remove_emojis      => false))

        sent_ids, vocab = prep.word_ids, prep.vocabulary
        ids = vcat(sent_ids...)                    # flatten sentences

        @test length(ids) >= 100             
        @test haskey(vocab.token2id, "jedi")      # 'jedi' still here

        model = WE.train!(ids, vocab;
                          epochs = 1,
                          emb_dim = 16,
                          batch  = 128,
                          rng    = MersenneTwister(314))

        emb = WE.embeddings(model)
        @test size(emb, 2) == length(vocab.id2token)
        @test all(isfinite, emb)                  # no NaNs / Infs
    end
end



@testset "vector helper returns correct slice - word" begin
    ids   = repeat(2:7, 200)                    # 1 200 tokens (words)
    vocab = make_vocab(6)                       # ids 2...7 map to "1"..."6"

    m = WE.train!(ids, vocab; epochs = 1, emb_dim = 8,
                  batch = 128, rng = MersenneTwister(7))

    E = WE.embeddings(m)
    for w in ["1","2","3","<unk>"]
        col = E[:, vocab.token2id[w]]
        @test col == WE.vector(m, vocab, w)
    end
end


@testset "nearest-neighbour sanity - word" begin
    using LinearAlgebra: dot, norm

    ids   = repeat(2:6, 250)                    # 1 250 tokens
    vocab = make_vocab(5)

    m  = WE.train!(ids, vocab; epochs = 1, emb_dim = 8,
                   batch = 128, rng = MersenneTwister(11))
    E  = WE.embeddings(m)

    cosine(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())

    tgt = vocab.token2id["1"]
    sims = [cosine(E[:,tgt], E[:,j]) for j in 1:size(E,2)]
    @test argmax(sims) == tgt                   # most similar to itself
end


@testset "sentence encoding helper - word" begin
    using LinearAlgebra: norm
    using Statistics: mean
    using TextSpace.Preprocessing: tokens_to_ids

    # tiny corpus and model
    ids   = repeat(2:6, 200)
    vocab = make_vocab(5)
    m     = WE.train!(ids, vocab; epochs = 1, emb_dim = 8,
                      batch = 128, rng = MersenneTwister(123))

    # two different in-vocab sentences
    sents   = ["1 1 2", "5 5 4 3"]
    id_seqs = [tokens_to_ids(String.(split(s)), vocab; add_new = false)
           for s in sents]

    # pad to same length (pad with <unk>=1)
    L = maximum(length, id_seqs)
    padvec(v) = vcat(v, fill(vocab.unk_id, L-length(v)))
    mat = hcat(padvec.(id_seqs)...)             # L x 2

    # average word embeddings over sequence length
    E   = WE.embeddings(m)
    vecs = E[:, mat]                            # 8 x L x 2 (view)
    sent_repr = dropdims(mean(vecs; dims = 2), dims = 2)'  # 2 x 8

    @test size(sent_repr) == (2, 8)
    @test all(isfinite, sent_repr)
    @test norm(sent_repr[1, :] .- sent_repr[2, :]) > 0      # vectors differ
end



@testset "user-workflow smoke-run (word)" begin
    PP = TextSpace.Preprocessing
    WE = TextSpace.WordEmbeddings

    # build a small corpus and preprocess it
    corpus = ["Hello world!", "Hello there,", "Good-bye world."]
    bigtxt = join(repeat(corpus, 40), ' ')          # around 180 words

    prep = PP.preprocess_for_word_embeddings(bigtxt;
                                             from_file   = false,
                                             min_count   = 1)
    sent_ids = prep.word_ids          # Vector{Vector{Int}}
    vocab    = prep.vocabulary

    flat_ids = vcat(sent_ids...)      # WE.train! wants a flat Vector{Int}

    @test length(flat_ids) >= 150
    @test haskey(vocab.token2id, "hello")

    # tiny Skip-Gram model (could use :cbow as well)
    model = WE.train!(flat_ids, vocab;
                      objective = :skipgram,
                      epochs    = 2,
                      emb_dim   = 16,
                      batch     = 128,
                      rng       = MersenneTwister(2025))

    E = WE.embeddings(model)
    @test norm(E[:, vocab.token2id["hello"]]) > 0     # non-zero vector

    # simple sentence encoder = mean of word embeddings
    encode(sent) = begin
        ids = PP.tokens_to_ids(String.(split(lowercase(sent))), vocab;
                               add_new = false)
        mean(E[:, ids]; dims = 2) |> vec
    end

    s1, s2, s3 = encode.(corpus)
    cosine(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())
    @test cosine(s1, s2) ≥ cosine(s1, s3)             # "Hello ..." sentences closer

    # save / reload round-trip
    tmp = tempname()*".tsv"
    WE.save_embeddings(tmp, model, vocab)

    # read first line back and compare with in-memory column
    first_tok, first_vec = let parts = split(readlines(tmp)[1], '\t')
        parts[1], Float32.(parse.(Float64, parts[2:end]))
    end
    col_vec = E[:, vocab.token2id[first_tok]]
    @test isapprox(first_vec, col_vec; atol = 1e-6)

    rm(tmp; force = true)
end


@testset "user-workflow big-picture (word)" begin
    PP = TextSpace.Preprocessing
    WE = TextSpace.WordEmbeddings
    using LinearAlgebra: dot, norm

    # ───────────────────────── 1. build a non-trivial corpus ─────────────────────
    raw_text = repeat("""
        Thoroughly conscious ignorance is the prelude to every real advance in science. James Clerk Maxwell.

        The important thing is to know how to take all things quietly. Michael Faraday.

        The secret we should never let the gamemasters know is that they don't need any rules. Gary Gygax.

        Squire Trelawney, Dr. Livesey, and the rest of these gentlemen having asked me to write down the whole
        particulars about Treasure Island, from the beginning to the end,
        keeping nothing back but the bearings of the island,
        and that only because there is still treasure not yet lifted, I take up my pen in the year of grace 17-,
        and go back to the time when my father kept the "Admiral Benbow" inn, and the brown old seaman, with the
        sabre cut, first took up his lodging under our roof. — Robert Louis Stevenson, *Treasure Island*.
        """, 6)                               # sufficient of tokens
        

    prep = PP.preprocess_for_word_embeddings(raw_text;
                                             from_file   = false,
                                             min_count   = 1)   # keep everything
    sent_ids  = prep.word_ids                       # Vector{Vector{Int}}
    vocab     = prep.vocabulary

    flat_ids  = vcat(sent_ids...)                   # training expects flat Vector
    @test length(flat_ids) ≥ 500                    # corpus large enough
    @test haskey(vocab.token2id, "science")         # a known word survives


    model = WE.train!(flat_ids, vocab;
                      objective = :skipgram,
                      epochs    = 3,
                      emb_dim   = 32,
                      batch     = 256,
                      rng       = MersenneTwister(4242))

    E = WE.embeddings(model)
    @test size(E, 2) == length(vocab.id2token)
    @test all(isfinite, E)


    cos(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())

    id_science = vocab.token2id["science"]
    id_rules   = vocab.token2id["rules"]
    @test cos(E[:, id_science], E[:, id_rules]) < 0.99   # they are not identical


    tmp = tempname()*".tsv"
    WE.save_embeddings(tmp, model, vocab)
    @test isfile(tmp)

    m2, v2 = WE.load_embeddings(tmp)
    @test E ≈ WE.embeddings(m2)
    @test vocab.token2id == v2.token2id
    rm(tmp; force = true)
end



#cbow

@testset "CBOW utilities - word" begin
    ids = collect(1:10)

    C,O = WU.build_word_pairs(ids;
                              mode   = :cbow,
                              win    = 5,
                              stride = 5,
                              radius = 2,
                              rng    = MersenneTwister(0))

    @test length(C) == length(O)
    @test all(length.(C) .== 4)          # 2⋅radius context words

    idx = findfirst(==(3), O)
    @test idx !== nothing

    ctx = C[idx]
    @test 1 in ctx && 5 in ctx && 3 ∉ ctx
end


@testset "CBOW training smoke-run - word" begin
    ids   = rand(2:7, 800)    # fake corpus (6 distinct words)
    vocab = make_vocab(6)

    m = WE.train!(ids, vocab;
                  objective = :cbow,
                  emb_dim   = 16,
                  batch     = 128,
                  epochs    = 1,
                  rng       = MersenneTwister(42))

    E = WE.embeddings(m)
    @test size(E) == (16, length(vocab.id2token))
    @test std(vec(E)) > 0
    @test all(isfinite, E)
end

@testset "deterministic CBOW - word" begin
    rng_pairs = MersenneTwister(99)
    ids   = rand(rng_pairs, 2:7, 1_000)
    vocab = make_vocab(6)

    Random.seed!(123)
    m1 = WE.train!(ids, vocab;
                   objective = :cbow,
                   epochs    = 1,
                   emb_dim   = 8,
                   rng       = copy(rng_pairs))

    Random.seed!(123)
    m2 = WE.train!(ids, vocab;
                   objective = :cbow,
                   epochs    = 1,
                   emb_dim   = 8,
                   rng       = copy(rng_pairs))

    @test WE.embeddings(m1) == WE.embeddings(m2)
end

@testset "CBOW loss drops - word" begin
    ids   = repeat(2:5, 2_500)          # 10 000 tokens (4 words)
    vocab = make_vocab(4)
    
    C,O = WU.build_word_pairs(ids; mode=:cbow,
                              win=20, stride=20, radius=1,
                              rng = MersenneTwister(11))
    C,O = C[1:256], O[1:256]

    k_neg   = 5
    neg_ctx = vcat(ntuple(_->C, k_neg)...)          # repeat ctx
    neg_ctr = rand(MersenneTwister(12),
                   1:length(vocab.id2token), length(O)*k_neg)

    m0 = WE.SkipGramModel(length(vocab.id2token), 16)
    l0 = WE.cbow_loss(m0, C, O, neg_ctx, neg_ctr)

    m1 = WE.train!(ids, vocab;
                   objective = :cbow,
                   epochs    = 6,
                   emb_dim   = 16,
                   batch     = 256,
                   lr        = 0.05,
                   rng       = MersenneTwister(321))

    l1 = WE.cbow_loss(m1, C, O, neg_ctx, neg_ctr)
    @test l1 < l0 - 1e-5             # improvement
end


@testset "CBOW user-workflow big-picture - word" begin
    PP = TextSpace.Preprocessing

    raw = """
        Thoroughly conscious ignorance is the prelude to every real advance in science. — James Clerk Maxwell
        The important thing is to know how to take all things quietly. — Michael Faraday
        The secret we should never let the game-masters know is that they don't need any rules. — Gary Gygax
        There is no absolute scale of size in the Universe, for it is boundless towards the great and also boundless towards the small. - Oliver Heaviside
        Why should I refuse a good dinner simply because I don't understand the digestive processes involved. - Oliver Heaviside
        … (same long paragraph) …
        """ |> lowercase |> x -> repeat(x, 10)

    prep = PP.preprocess_for_word_embeddings(raw; from_file=false, min_count=1)
    sent_ids, vocab = prep.word_ids, prep.vocabulary
    flat_ids = vcat(sent_ids...)

    @test length(flat_ids) >= 500
    @test haskey(vocab.token2id, "science")   # choose a word that’s there!

    model = WE.train!(flat_ids, vocab;
                      objective = :cbow,
                      epochs    = 3,
                      emb_dim   = 32,
                      batch     = 256,
                      rng       = MersenneTwister(4242))

    E = WE.embeddings(model)
    @test size(E,2) == length(vocab.id2token)
    @test all(isfinite, E)

    cos(a,b) = dot(a,b)/(norm(a)*norm(b)+eps())
    id_a = vocab.token2id["science"]
    id_b = vocab.token2id["rules"]
    @test cos(E[:, id_a], E[:, id_b]) < 0.99   # not identical
end




