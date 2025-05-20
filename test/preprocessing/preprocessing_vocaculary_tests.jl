include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Vocabulary.jl"))


@testset "build_vocabulary — happy path" begin
    toks = ["hello", "world", "hello", "foo"]
    vdict = build_vocabulary(toks;                    # <- function under test
                             min_freq      = 1,
                             special_tokens = ["<pad>", "<unk>"])

    ti  = vdict["token_to_index"]
    it  = vdict["index_to_token"]
    freq = vdict["freq"]

    @test it[1] == "<pad>"           # special tokens are first
    @test it[2] == "<unk>"
    @test ti["hello"] == 3           # after 2 specials, 'hello' is third
    @test freq["hello"] == 2         # counted twice
    @test length(it) == 5            # <pad> <unk> hello world foo
end

@testset "convert_tokens_to_ids ↔ ids_to_tokens round-trip" begin
    voc = Vocabulary(Dict("<unk>" => 1),        # token2id
                     ["<unk>"],                 # id2token
                     Dict{Int,Int}(),           # counts
                     1)                         # unk_id

    ids = convert_tokens_to_ids(["foo","foo","bar"], voc;
                                add_new = true, update_counts = false)

    @test ids == [2,2,3]
    @test voc.id2token == ["<unk>","foo","bar"]

    toks_back = convert_ids_to_tokens(ids, voc)
    @test toks_back == ["foo","foo","bar"]
end



@testset "min_freq filters singletons" begin
    toks  = ["a","b","a","c","d","d","d"]         # b and c appear once
    vdict = build_vocabulary(toks; min_freq = 2)

    @test !haskey(vdict["token_to_index"], "b")   # filtered out
    @test !haskey(vdict["token_to_index"], "c")
    @test vdict["freq"]["d"] == 3                 # freq table is still full
end


@testset "max_vocab_size truncates after sorting by freq" begin
    toks  = ["w","x","y","z","w","x","y"]         # w=2, x=2, y=2, z=1
    vdict = build_vocabulary(toks;
                             max_vocab_size = 2)  # keep the 2 most frequent

    kept = keys(vdict["token_to_index"])
    @test length(kept) == 2
    @test all(k ∈ ("w","x","y") for k in kept)    # z cannot be present
end


@testset "special tokens are unique and kept in order" begin
    toks  = ["foo","bar"]
    vdict = build_vocabulary(toks;
                             special_tokens = ["<pad>","<unk>","<pad>"]) # duplicate

    it = vdict["index_to_token"]
    @test it[1:2] == ["<pad>","<unk>"]   # duplicate removed
    @test count(==( "<pad>" ), it) == 1  # only once in the final list
end


@testset "convert_tokens_to_ids with add_new = false (OOV logic)" begin
    # Build a tiny fixed vocab first
    voc = Vocabulary(Dict("foo"=>1,"bar"=>2,"<unk>"=>3),
                     ["foo","bar","<unk>"],
                     Dict{Int,Int}(),
                     3)            # unk_id = 3

    ids = convert_tokens_to_ids(["foo","baz"], voc;
                                add_new = false, update_counts = false)

    @test ids == [1,3]                       # baz mapped to unk_id
    @test !haskey(voc.token2id, "baz")       # vocabulary unchanged
end


@testset "counts are updated correctly" begin
    voc = Vocabulary(Dict("<unk>" => 1),
                     ["<unk>"],
                     Dict{Int,Int}(),
                     1)

    _ = convert_tokens_to_ids(["foo","foo","bar"], voc;
                              add_new = true, update_counts = true)

    @test voc.counts[2] == 2   # foo seen twice
    @test voc.counts[3] == 1   # bar seen once
    @test !haskey(voc.counts, 1) # <unk> never used -> no entry
end


@testset "convert_batch_tokens_to_ids pads to max length" begin
    voc = Vocabulary(Dict("<unk>" => 1),
                     ["<unk>"],
                     Dict{Int,Int}(),
                     1)

    batch = [["a","b","c","d"], ["e","f"]]
    M = convert_batch_tokens_to_ids(batch, voc;
                                    add_new = true,
                                    update_counts = false)

    @test size(M) == (4, 2)                 # longest doc length × n_docs
    @test all(M[3:end, 2] .== 1)            # rows 3-4 of column 2 padded with unk_id
end



@testset "save_vocabulary -> load_vocabulary round-trip" begin
    vdict = build_vocabulary(["α","β","β","γ"])
    tmp   = tempname()

    save_vocabulary(vdict, tmp)
    vdict2 = load_vocabulary(tmp)

    @test vdict2 == vdict              # JSON round-trip exact
end
