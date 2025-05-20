include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "ParagraphProcessing.jl"))


# handy dummy tokenizer: counts whitespace-separated tokens
dummy_tok(s) = split(normalize_whitespace(s))

@testset "unwrap_lines" begin
    txt = "This is a line\nbroken in the middle.\nBut this ends.\n\nNext."
    out = unwrap_lines(txt)
    @test occursin("line broken", out)          # newline folded to space
    @test occursin(".\n Next", out)            # double NL kept (paragraph end)
end

@testset "split_paragraphs" begin
    raw = "First para line 1\nline 2.\n\n Second para."
    ps  = split_paragraphs(raw)                 # unwrap + normalize default
    @test length(ps) == 2
    @test ps[1] == "First para line 1 line 2."
    @test ps[2] == "Second para."

    ps2 = split_paragraphs(raw; unwrap = false, normalize = false)
    @test ps2[1] == "First para line 1\nline 2."
end

@testset "filter_paragraphs" begin
    long  = "a"^30
    short = "tiny"
    kept  = filter_paragraphs([long, short]; min_chars = 10)
    @test kept == [long]
end

@testset "paragraph_windows" begin
    paras = ["alpha beta", "gamma delta epsilon", "zeta"]
    win   = paragraph_windows(paras, 4; stride = 2, tokenizer = dummy_tok)

    chunks = Vector{Vector{String}}()
    itr, st = win
    while true
        res = itr(st)
        res === nothing && break
        chunk, st = res
        push!(chunks, chunk)
    end

    @test chunks[1] == ["alpha beta"]      # first paragraph fits (2 tokens)
    @test chunks[2] == ["zeta"]            # stride = 2 skips to 3rd paragraph
end


@testset "char_span_to_paragraph_idx" begin
    txt   = "Para1.\n\nPara2 longer.\n\nLast."
    paras = String.(split_paragraphs(txt; unwrap=false, normalize=false))  # promote

    start_para2 = first(findfirst("Para2", txt))
    idxs = char_span_to_paragraph_idx([(start_para2, start_para2 + 4)], paras)

    @test idxs == [2]                       # second paragraph
end
