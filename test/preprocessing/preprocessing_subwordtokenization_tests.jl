include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SubwordTokenization.jl"))


need_bpe = false
try
    @eval using BytePairEncoding
    global need_bpe = isdefined(BytePairEncoding, :learn_bpe)   # disambiguate
catch
    # package not installed, leave need_bpe = false
end

@testset "Subword tokenisation (BPE)" begin
    if !need_bpe
        @info "BytePairEncoding.learn_bpe not available - skipping BPE tests"
        @test true               # keep the set green
        return
    end

    using Serialization

    #tiny corpus
    corpus_path = tempname()
    open(corpus_path, "w") do io
        println(io, "hello world")
        println(io, "hello there world")
    end

    model_path = tempname()

    
    #train  ->  serialize  ->  load
    file_out = train_bpe([corpus_path];
                         vocab_size = 64,
                         num_merges = 100,
                         model_path = model_path)

    @test file_out == model_path && isfile(model_path)

    tok = load_bpe(model_path)
    @test all(sym in keys(tok.vocab) for sym in DEFAULT_SPECIAL_TOKENS)

    #encode / decode round-trip
    txt  = "hello world"
    ids  = encode(tok, txt)
    @test decode(tok, ids) == txt

    ids_sp = encode(tok, txt; add_special_tokens = true)
    @test first(ids_sp) == special_id(tok, "<cls>")
    @test last(ids_sp)  == special_id(tok, "<sep>")
    @test length(ids_sp) == length(ids) + 2

    #batch matrix + padding
    pad  = special_id(tok, "<pad>")
    mat  = encode_batch(tok, ["hi", "hello"])
    @test size(mat, 2) == 2
    @test mat[end,1] == pad
    @test mat[1:length(encode(tok,"hi")),1] == encode(tok,"hi")

    #vocabulary accessor
    v = vocabulary(tok)
    @test isa(v, Dict{String,Int}) && v["<unk>"] isa Int
end
