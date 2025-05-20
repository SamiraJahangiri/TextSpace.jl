using Test, Random
using TextSpace                  # pulls in CharacterEmbeddings
using Statistics

const CE = TextSpace.CharacterEmbeddings
const V  = TextSpace.Preprocessing.Vocabulary
const PP = TextSpace.Preprocessing


@testset "Character-Embeddings - training smoke-run" begin
    
    ids = rand(1:6, 600) # fake corpus with 6 chars
    tokmap = Dict("<unk>" => 1, "1"=>2, "2"=>3, "3"=>4, "4"=>5, "5"=>6, "6"=>7)
    id2tok = Vector{String}(undef, length(tokmap))
    for (tok, id) in tokmap
        id2tok[id] = tok
    end
    vocab = Vocabulary(tokmap, id2tok, Dict{Int,Int}(), 1)


    model = CE.train!(ids, vocab;
                      emb_dim = 16,
                      batch   = 128,
                      epochs  = 1,
                      k_neg   = 2,
                      lr      = 1e-1,
                      rng     = MersenneTwister(42))


    E = CE.embeddings(model)   # (16 × 7) matrix
    @test size(E) == (16, length(vocab.id2token))

    # embeddings should not all be identical after one update
    @test Statistics.std(vec(E)) > 0.0

    # saving works and file exists
    tmp = tempname()*".tsv"
    CE.save_embeddings(tmp, model, vocab)
    @test isfile(tmp)
    rm(tmp; force=true)
end


@testset "deterministic Skip-Gram" begin
    rng_pairs = MersenneTwister(42)
    ids       = rand(rng_pairs, 1:8, 2_000)

    #basic vocabulary 
    tok2id = Dict(string(i)=>i for i in 1:8);  tok2id["<unk>"] = 9
    id2tok = [string.(1:8)..., "<unk>"]
    vocab  = V(tok2id, id2tok, Dict{Int,Int}(), 9)

    # data 1 
    Random.seed!(123)
    m1 = CE.train!(ids, vocab;
                   epochs = 1,
                   lr     = 0.01,          # smaller LR, learning rate eta
                   batch  = 256,
                   k_neg  = 2,
                   rng    = copy(rng_pairs))

    # data 2 
    Random.seed!(123)
    m2 = CE.train!(ids, vocab;
                   epochs = 1,
                   lr     = 0.01,
                   batch  = 256,
                   k_neg  = 2,
                   rng    = copy(rng_pairs))

    @test isequal(CE.embeddings(m1), CE.embeddings(m2))    # deterministic
    @test !any(isnan, CE.embeddings(m1))                   # finite values
end



@testset "loss drops" begin
    #tiny corpus & vocabulary (ids 1to5 + <unk>=6)
    ids     = repeat(1:5, 200)
    tok2id  = Dict(string(i)=>i for i in 1:5);  tok2id["<unk>"] = 6
    vocab   = V(tok2id, [string.(1:5)..., "<unk>"], Dict{Int,Int}(), 6)

    #fixed probe batch: 50 skip-gram pairs + negatives
    C, O = CE.build_char_pairs(ids; mode=:skipgram, win=20, stride=20,
                            radius=1, rng = MersenneTwister(77))
    sel  = 1:50;   C, O = C[sel], O[sel]

    rng_neg = MersenneTwister(88)
    kneg    = 5
    neg_O   = rand(rng_neg, 1:length(vocab.id2token), length(O)*kneg)
    neg_C   = repeat(C, kneg)

    #baseline loss with reproducible initial weights
    Random.seed!(321)
    model0 = CE.SkipGramModel(length(vocab.id2token), 16)
    l1 = CE.sg_loss(model0, C, O, neg_C, neg_O)

    #train for 5 epochs starting from the same init
    Random.seed!(321)
    model = CE.train!(ids, vocab;
                      epochs = 10,
                      emb_dim = 16,
                      lr     = 0.02,
                      batch  = 128,
                      rng    = MersenneTwister(1))

    l2 = CE.sg_loss(model, C, O, neg_C, neg_O)

    @test l2 < l1 #averaged loss must decrease
end


@testset "loss improves with more epochs" begin
    #synthetic corpus (10_000 tokens, ids 1to5)
    big_ids  = repeat(1:5, 2_000)            # 10 000 tokens
    tok2id   = Dict(string(i)=>i for i in 1:5);  tok2id["<unk>"] = 6
    vocab6   = V(tok2id, [string.(1:5)...,"<unk>"], Dict{Int,Int}(), 6)

    #fixed probe batch (same idea as previous test)
    using TextSpace.CharacterEmbeddings.CharacterEmbeddingUtilities: build_char_pairs
    Cp, Op = build_char_pairs(big_ids; mode=:skipgram, win=40, stride=40,
                              radius=2, rng=MersenneTwister(90))
    Cp, Op = Cp[1:100], Op[1:100]    # 100 probe pairs
    kneg   = 5
    neg_O  = rand(MersenneTwister(91),
                  1:length(vocab6.id2token), length(Op)*kneg)
    neg_C  = repeat(Cp, kneg)

    #train 3 epochs, then 6 epochs from the SAME initial weights
    Random.seed!(555)
    model3 = CE.train!(big_ids, vocab6;
                       epochs = 3, emb_dim = 32, lr = 0.02,
                       batch  = 256, rng = MersenneTwister(1))

    Random.seed!(555)       # identical init again
    model6 = CE.train!(big_ids, vocab6;
                       epochs = 6, emb_dim = 32, lr = 0.02,
                       batch  = 256, rng = MersenneTwister(1))

    l3 = CE.sg_loss(model3, Cp, Op, neg_C, neg_O)
    l6 = CE.sg_loss(model6, Cp, Op, neg_C, neg_O)

    @test l6 <= l3  # six epochs never worse than three
end



@testset "save / reload" begin
    #toy corpus & vocabulary
    ids   = repeat(1:3, 100)                     # 300 tokens, ids in 1:3
    tok2id = Dict("1"=>1, "2"=>2, "3"=>3, "<unk>"=>4)
    id2tok = ["1","2","3","<unk>"]
    vocab  = V(tok2id, id2tok, Dict{Int,Int}(), 4)

    #train a  single-epoch Skip-Gram model
    model = CE.train!(ids, vocab; epochs = 1, emb_dim = 8, batch = 64)

    #save and reload the embedding matrix
    tmp = tempname()*".tsv"
    CE.save_embeddings(tmp, model, vocab)

    #read file: drop first column (token) and collect the vectors
    cols = (parse.(Float64, split(line, '\t')[2:end]) for line in eachline(tmp))
    E = hcat(cols...) |> Matrix # emb_dim × vocab_size

    @test E ≈ CE.embeddings(model)  # round-trip exact
    rm(tmp; force = true)
end


@testset "OOV triggers BoundsError" begin
    #model with vocab_size = 4  (ids 1to4)
    model = CE.SkipGramModel(4, 8)  # emb_dim = 8

    # id 5 is outside the valid range => should raise BoundsError
    @test_throws BoundsError CE.dot_scores(model, [1], [5])
end


function safe_idx(id, vocab_size, unk_id)
    return 1 <= id <= vocab_size ? id : unk_id
end

function dot_scores(m::CE.SkipGramModel, centers, contexts; unk_id = size(m.emb.weight, 2))
    safe_idx(x) = 1 ≤ x ≤ size(m.emb.weight,2) ? x : unk_id
    ctx = map(safe_idx, contexts)
    zc  = m.emb(centers)
    zo  = m.emb(ctx)
    return vec(sum(zc .* zo; dims = 1))
end

@testset "dot_scores rejects OOV ids" begin
    m = CE.SkipGramModel(4, 8)        # vocab size = 4
    @test_throws BoundsError CE.dot_scores(m, [1], [5])
end


@testset "batch boundary safety" begin
    ids   = repeat(1:4, 130)                    # 520 tokens
    tok2id = Dict(string(i)=>i for i in 1:4); tok2id["<unk>"]=5
    vocab  = V(tok2id, ["1","2","3","4","<unk>"], Dict{Int,Int}(), 5)

    # batch=256 will force one full batch plus one short batch
    model = CE.train!(ids, vocab; epochs=1, batch=256, lr=0.02, emb_dim=8)

    @test !any(isnan, CE.embeddings(model))
end



using TextSpace.Preprocessing: preprocess_for_char_embeddings

@testset "pipeline smoke-run" begin
    #make the raw text long enough (>128 chars)
    raw = repeat("Hello 😊 — hello! ", 10)        # around 180 chars

    out   = preprocess_for_char_embeddings(raw)
    ids   = out.char_ids
    vocab = out.vocabulary

    @test length(ids) > 128                      # enough for pair builder
    @test haskey(vocab.token2id, "H")            # capital H is in vocab

    # 2.  One-epoch training should run without NaNs
    model = CE.train!(ids, vocab;
                      epochs = 1,
                      emb_dim = 8,
                      batch  = 32,   #small batch for speed
                      rng    = MersenneTwister(123))

    E = CE.embeddings(model)
    @test size(E, 2) == length(vocab.id2token)   # correct width
    @test !any(isnan, E)              #finite embeddings
end

@testset "negative-sampling sweep" begin
    ids    = repeat(1:4, 400)                 # 4times400 = 1600 tokens
    tok2id = Dict(string(i)=>i for i in 1:4); tok2id["<unk>"]=5
    vocab  = V(tok2id, ["1","2","3","4","<unk>"], Dict{Int,Int}(), 5)

    for k in 1:6
        m = CE.train!(ids, vocab;
                      epochs = 1,
                      emb_dim = 8,
                      batch  = 128,
                      k_neg  = k,
                      rng    = MersenneTwister(100+k))
        @test !any(isnan, CE.embeddings(m))
    end
end

@testset "batch size = 1 edge-case" begin
    ids    = rand(1:5, 150)                  # 150 tokens
    tok2id = Dict(string(i)=>i for i in 1:5); tok2id["<unk>"]=6
    vocab  = V(tok2id, [string.(1:5)...,"<unk>"], Dict{Int,Int}(), 6)

    m = CE.train!(ids, vocab;
                  epochs = 1,
                  emb_dim = 8,
                  batch  = 1,                # single-sample batches
                  rng    = MersenneTwister(7))

    @test !any(isnan, CE.embeddings(m))
end


@testset "Unicode corpus" begin
    # 200 plus characters, plenty of multi-byte codepoints
    txt = repeat("🎉 café 🐍 ", 25)            # about 225 visible chars

    mktemp() do path, io
        write(io, txt)                        # save corpus to a short path
        close(io)

        out = preprocess_for_char_embeddings(path)  # <- file-input branch
        @test length(out.char_ids) > 128            # enough for windows
        @test haskey(out.vocabulary.token2id, "🎉") # emoji survived

        m = CE.train!(out.char_ids, out.vocabulary;
                      epochs = 1,
                      emb_dim = 8,
                      batch  = 64,
                      rng    = MersenneTwister(99))

        @test !any(isnan, CE.embeddings(m))         # finite embeddings
    end
end



@testset "real-text corpus smoke-run (French)" begin
    #from https://www.amazon.com/Moisson-rouge/dp/2266268082
    excerpt = """
        Aux heures les plus noires de l'Ancienne République, alors que les 
        Chevaliers Jedi combattent les Seigneurs Sith et leurs armées impitoyables, 
        Darth Scabrous poursuit son rêve fanatique sur le point de devenir une réalité cauchemardesque.
        Parmi les Jedi du Corps Agricole, Hestizo Trace ppossède un extraordinaire talent: 
        un don avec les plantes qui lui a permis d'élever sa chère et précieuse orchidée noire. 
        Une fleur rare dont doit s'emparer à tout prix l'émissaire de Darth Scabrous. 
        Car elle est l'ingrédient final d'une préparation funeste, censée offrir l'immortalité.
        """

    mktemp() do path, io
        write(io, excerpt)
        close(io)

        out = preprocess_for_char_embeddings(path)
        ids, vocab = out.char_ids, out.vocabulary

        @test length(ids) >= 300                    # > one full window
        @test haskey(vocab.token2id, "é")          # accented char kept

        model = CE.train!(ids, vocab;
                          epochs = 1,
                          emb_dim = 16,
                          batch  = 128,
                          rng    = MersenneTwister(314))

        emb = CE.embeddings(model)
        @test size(emb, 2) == length(vocab.id2token)
        @test all(isfinite, emb)                   # no NaNs / Infs
    end
end



@testset "real-text corpus smoke-run" begin
    # from https://ew.com/star-wars-the-living-force-exclusive-excerpt-qui-gon-obi-wan-8610658
    excerpt = """ 
        The Jedi have always traveled the stars, defending peace and justice across the galaxy. 
        But the galaxy is changing, and the Jedi Order along with it. 
        More and more, the Order finds itself focused on the future of the Republic, secluded on Coruscant, 
        where the twelve members of the Jedi Council weigh crises on a galactic scale.
        As yet another Jedi Outpost left over from the Republic's golden age is set to be decommissioned on the 
        planet Kwenn, Qui-Gon Jinn challenges the Council about the Order's increasing isolation. 
        Mace Windu suggests a bold response: All twelve Jedi Masters will embark on a goodwill 
        mission to help the planet and to remind the people of the galaxy that the 
        Jedi remain as stalwart and present as they have been across the ages.
        """

    mktemp() do path, io
        write(io, excerpt)
        close(io)

        out = preprocess_for_char_embeddings(path)
        ids, vocab = out.char_ids, out.vocabulary

        @test length(ids) >= 300                    # plenty of characters
        @test haskey(vocab.token2id, "J")          # capital letter kept

        model = CE.train!(ids, vocab;
                          epochs = 1,
                          emb_dim = 16,
                          batch  = 128,
                          rng    = MersenneTwister(314))

        emb = CE.embeddings(model)
        @test size(emb, 2) == length(vocab.id2token)
        @test all(isfinite, emb)                   # no NaNs / Infs
    end
end


@testset "vector helper returns correct slice" begin
    ids   = repeat(1:4, 200)                       # 800 chars
    tok2id = Dict(string(i)=>i for i in 1:4); tok2id["<unk>"]=5
    vocab  = V(tok2id, ["1","2","3","4","<unk>"], Dict{Int,Int}(), 5)

    model = CE.train!(ids, vocab; epochs=1, emb_dim=8, batch=128,
                      rng=MersenneTwister(7))

    for ch in ["1","2","3","<unk>"]
        col = CE.embeddings(model)[:, vocab.token2id[ch]]
        @test col == CE.vector(model, vocab, ch)
    end
end

@testset "nearest-neighbour sanity" begin
    using LinearAlgebra: dot, norm 

    # throw-away tiny model
    ids    = repeat(1:4, 200)
    tok2id = Dict(string(i)=>i for i in 1:4);  tok2id["<unk>"] = 5
    vocab  = V(tok2id, ["1","2","3","4","<unk>"], Dict{Int,Int}(), 5)

    model = CE.train!(ids, vocab; epochs = 1, emb_dim = 8,
                      batch  = 128, rng = MersenneTwister(11))

    emb = CE.embeddings(model)
    cosine(a,b) = dot(a,b) / (norm(a) * norm(b) + eps())

    sims = [cosine(emb[:,1], emb[:,j]) for j in 1:size(emb,2)]
    @test argmax(sims) == 1                  # self-similarity highest
end


@testset "sentence encoding helper" begin
    using LinearAlgebra: norm
    using Statistics: mean
    using TextSpace.Preprocessing:
          tokenize_char, chars_to_ids, pad_sequences
    
    ids      = repeat(1:4, 100)
    tok2id   = Dict(string(i)=>i for i in 1:4);  tok2id["<unk>"] = 5
    vocab    = V(tok2id, ["1","2","3","4","<unk>"], Dict{Int,Int}(), 5)

    model = CE.train!(ids, vocab; epochs=1, emb_dim=8,
                      batch=128, rng=MersenneTwister(123))

    #encode two DIFFERENT in-vocab sentences
    sentences = ["1112", "4443"]            #these digits are in vocab
    id_seqs   = [chars_to_ids(tokenize_char(s), vocab) for s in sentences]
    mat       = pad_sequences(id_seqs; pad_value=vocab.unk_id)   # L x 2

    vecs      = CE.embeddings(model)[:, mat]          # 8 x L x 2
    sent_repr = dropdims(mean(vecs; dims=2), dims=2)'  # 2 x 8

    @test size(sent_repr) == (2, 8)
    @test all(isfinite, sent_repr)
    @test norm(sent_repr[1, :] .- sent_repr[2, :]) > 0   # now different
end



@testset "user-workflow smoke-run" begin
    PP = TextSpace.Preprocessing
    corpus = ["Hello world!", "Hello there,", "Good-bye world."]

    #  make a >128-character corpus and store it in a temp file 
    bigtxt = join(repeat(corpus, 20), ' ')     # 360 chars
    ids, vocab = let
        mktemp() do path, io
            write(io, bigtxt); close(io)                # write once
            out = PP.preprocess_for_char_embeddings(path)
            out.char_ids, out.vocabulary
        end
    end

    @test length(ids) ≥ 128
    @test haskey(vocab.token2id, "H")

    #  train a tiny model 
    model = CE.train!(ids, vocab; epochs=2, emb_dim=16,
                      batch=128, rng=MersenneTwister(2025))
    emb   = CE.embeddings(model)

    # single-character lookup
    @test norm(emb[:, vocab.token2id["H"]]) > 0

    # sentence encoder (mean of character vectors)
    encode(sent) = mean(emb[:, PP.chars_to_ids(PP.tokenize_char(sent), vocab)];
                         dims=2) |> vec
    s1, s2, s3 = encode.(corpus)
    cosine(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())
    @test cosine(s1, s2) >= cosine(s1, s3)      # Hello  closer to each other

    # save / reload round-trip 
    tmp = tempname()*".tsv"
    CE.save_embeddings(tmp, model, vocab)

    parsed = [(split(l, '\t')[1],
               parse.(Float64, split(l, '\t')[2:end])) for l in eachline(tmp)]
    first_tok, first_vec = parsed[1]
    col_vec = emb[:, vocab.token2id[first_tok]]

    @test isapprox(vec(first_vec), vec(col_vec); atol=1e-6)

    rm(tmp; force=true)
end


@testset "user-workflow big-picture" begin
    PP = TextSpace.Preprocessing

    raw_text = repeat("""
        Thoroughly conscious ignorance is the prelude to every real advance in science. James Clerk Maxwell 

        The important thing is to know how to take all things quietly. Michael Faraday

        The secret we should never let the gamemasters know is that they don't need any rules. Gary Gygax

        Squire Trelawney, Dr. Livesey, and the rest of these gentlemen having asked me to write down the whole
        particulars about Treasure Island, from the beginning to the end, 
        keeping nothing back but the bearings of the island, 
        and that only because there is still treasure not yet lifted, I take up my pen in the year of grace 17-, 
        and go back to the time when my father kept the "Admiral Benbow" inn, and the brown old seaman, with the 
        sabre cut, first took up his lodging under our roof. Robert Louis Stevenson, Treasure Island
        """, 5)                          # 1 kB - plenty of characters


        ids, vocab = let
        mktemp() do path, io
            write(io, raw_text); close(io)             # save corpus once
            out = PP.preprocess_for_char_embeddings(raw_text; from_file=false)
            out.char_ids, out.vocabulary
        end
    end

    @test length(ids) >= 500
    @test haskey(vocab.token2id, ",")


    model = CE.train!(ids, vocab;
                      epochs = 3,
                      emb_dim = 32,
                      batch  = 256,
                      rng    = MersenneTwister(4242))

    emb = CE.embeddings(model)
    @test size(emb, 2) == length(vocab.id2token)
    @test all(isfinite, emb)


    cos(a,b) = dot(a,b)/(norm(a)*norm(b)+eps())
    id_comma = vocab.token2id[","]
    id_dot   = vocab.token2id["."]
    @test cos(emb[:, id_comma], emb[:, id_dot]) < 0.99   # commas not equal stops
end


#---------------------
#CBOW tests
make_vocab(N) = let
    tok2id = Dict(string(i)=>i for i in 1:N)
    tok2id["<unk>"] = N + 1
    V(tok2id, [string.(1:N)..., "<unk>"], Dict{Int,Int}(), N + 1)
end

@testset "CBOW utilities" begin
    ids = 1:10 |> collect

    C, O = CE.build_char_pairs(ids;
                               mode   = :cbow,
                               win    = 5,
                               stride = 5,
                               radius = 2,
                               rng    = MersenneTwister(0))

    @test length(C) == length(O)               # same number of tuples
    @test all(length.(C) .== 4)                # 2·radius context tokens

    # find the tuple whose centre is 3
    idx = findfirst(==(3), O)
    @test idx !== nothing                     # such a pair must exist

    ctx = C[idx]
    @test 1 in ctx && 5 in ctx && 3 ∉ ctx     # context = [1,2,4,5] in some order
end

@testset "CBOW training smoke-run" begin
    ids   = rand(1:6, 800)                       # small fake corpus
    vocab = make_vocab(6)

    model = CE.train!(ids, vocab;
                      objective = :cbow,
                      emb_dim   = 16,
                      batch     = 128,
                      epochs    = 1,
                      rng       = MersenneTwister(42))

    E = CE.embeddings(model)
    @test size(E) == (16, length(vocab.id2token))
    @test Statistics.std(vec(E)) > 0             # not all equal / zero
    @test all(isfinite, E)
end

@testset "deterministic CBOW" begin
    rng_pairs = MersenneTwister(99)
    ids       = rand(rng_pairs, 1:5, 1_000)
    vocab     = make_vocab(5)

    Random.seed!(123)
    m1 = CE.train!(ids, vocab;
                   objective = :cbow,
                   epochs = 1, emb_dim = 8, rng = copy(rng_pairs))

    Random.seed!(123)
    m2 = CE.train!(ids, vocab;
                   objective = :cbow,
                   epochs = 1, emb_dim = 8, rng = copy(rng_pairs))

    @test isequal(CE.embeddings(m1), CE.embeddings(m2))
end


@testset "CBOW loss drops" begin
    ids   = repeat(1:4, 250)                      # tiny corpus (1 000 tokens)
    vocab = make_vocab(4)

    #  fixed probe batch
    k_neg = 5                                     # negatives per pair
    C, O  = CE.build_char_pairs(ids; mode=:cbow,
                                win=20, stride=20, radius=1,
                                rng = MersenneTwister(11))
    C, O  = C[1:64],  O[1:64]                     # 64 probe pairs

    # corresponding negative samples (same ctx, random centre ids)
    rng_neg  = MersenneTwister(12)
    neg_ctx  = vcat(ntuple(_->C, k_neg)...)       # repeat contexts k_neg times
    neg_ctr  = rand(rng_neg, 1:length(vocab.id2token), length(O)*k_neg)

    #  baseline loss
    Random.seed!(321)
    init = CE.SkipGramModel(length(vocab.id2token), 16)
    l0   = CE.cbow_loss(init, C, O, neg_ctx, neg_ctr)

    #  train for a few epochs
    Random.seed!(321)              # same initial weights as `init`
    trained = CE.train!(ids, vocab;
                        objective = :cbow,
                        epochs    = 5,
                        emb_dim   = 16,
                        batch     = 128,
                        rng       = MersenneTwister(321))

    l1 = CE.cbow_loss(trained, C, O, neg_ctx, neg_ctr)

    @test l1 < l0                  # training should lower the (probe) loss
end



@testset "CBOW - user-workflow big-picture" begin
    PP = TextSpace.Preprocessing

    #corpus (≈1 kB after the repeat)
    raw_text = repeat("""
        Thoroughly conscious ignorance is the prelude to every real advance in science. — James Clerk Maxwell

        The important thing is to know how to take all things quietly. — Michael Faraday

        The secret we should never let the game-masters know is that they don't need any rules. — Gary Gygax

        Squire Trelawney, Dr. Livesey, and the rest of these gentlemen having asked me to write down the whole
        particulars about Treasure Island, from the beginning to the end,
        keeping nothing back but the bearings of the island, and that only because there is still treasure not
        yet lifted, I take up my pen in the year of grace 17-, and go back to the time when my father kept the
        'Admiral Benbow' inn, and the brown old seaman, with the sabre-cut, first took up his lodging under
        our roof. — Robert Louis Stevenson, *Treasure Island*
        """, 5)           

    # preprocessing directly from the string (from_file = false) 
    out   = PP.preprocess_for_char_embeddings(raw_text; from_file = false)
    ids   = out.char_ids
    vocab = out.vocabulary

    @test length(ids) ≥ 500
    @test haskey(vocab.token2id, ",")

    #CBOW training (3 epochs, 32-dim)
    model = CE.train!(ids, vocab;
                      objective = :cbow,
                      epochs    = 3,
                      emb_dim   = 32,
                      batch     = 256,
                      rng       = MersenneTwister(4242))

    emb = CE.embeddings(model)
    @test size(emb, 2) == length(vocab.id2token)
    @test all(isfinite, emb)

    #trivial semantic sanity check: ',' vs '.' are not identical
    cos(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())
    id_comma = vocab.token2id[","]
    id_dot   = vocab.token2id["."]
    @test cos(emb[:, id_comma], emb[:, id_dot]) < 0.99
end






