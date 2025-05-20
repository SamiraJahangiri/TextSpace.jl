

@testset "basic_tokenize" begin
    txt = "Hello,  World!\n"

    t = basic_tokenize(txt)
    @test t == ["Hello", ",", "World", "!"]

    t_ws = basic_tokenize(txt; keep_whitespace = true)
    # there should be exactly two extra tokens: the run of spaces and the newline
    @test length(t_ws) == length(t) + 2

    # confirm we captured a pure-whitespace token and the newline token
    @test any(t -> occursin(WHITESPACE_REGEX, t) && t ≠ "\n", t_ws)  # spaces
    @test any(==("\n"), t_ws)                                         # newline
end


@testset "tokenize end-to-end" begin
    @test tokenize("Hello, World!") == ["hello", "world"]   # default pipeline
    @test tokenize("This is a test.";
                   remove_stopwords = true) == ["this", "test"]

    # bigrams
    bigr = tokenize("a b c d"; ngram = 2, strip_punctuation = false, lower = false)
    @test bigr == ["a_b", "b_c", "c_d"]
end


@testset "tokens_to_ids" begin
    voc = Vocabulary(Dict("<unk>" => 1, "foo" => 2),
                     ["<unk>", "foo"],
                     Dict{Int,Int}(), 1)

    # unknown token maps to unk_id when add_new = false
    ids1 = tokens_to_ids(["foo", "bar"], voc; add_new = false)
    @test ids1 == [2, 1]

    # add_new = true inserts bar
    ids2 = tokens_to_ids(["foo", "bar"], voc; add_new = true)
    @test ids2 == [2, 3]
    @test voc.id2token[3] == "bar"
end

@testset "docs_to_matrix" begin
    voc = Vocabulary(Dict("<unk>" => 1, "x"=>2, "y"=>3),
                     ["<unk>", "x", "y"],
                     Dict{Int,Int}(), 1)

    docs = [["x","y","x"], ["y"]]
    mat  = docs_to_matrix(docs, voc)

    @test size(mat) == (3, 2)
    @test mat[:,1] == [2, 3, 2]              # x y x
    @test all(mat[2:3, 2] .== 1)             # padding with unk_id
end


