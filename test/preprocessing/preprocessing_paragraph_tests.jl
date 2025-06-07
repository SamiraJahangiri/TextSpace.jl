using Test
using TextSpace.Preprocessing:
    unwrap_lines,
    split_paragraphs,
    drop_empty_paragraph,
    merge_short_paragraphs,
    paragraph_windows,
    _is_blank_paragraph,
    filter_paragraphs

const PP = TextSpace.Preprocessing



@testset "unwrap_lines" begin
    txt = "This is a line\nbroken in the middle.\nBut this ends.\n\nNext."
    out = unwrap_lines(txt)

    @test occursin("line broken", out)           # hard-wrap folded
    @test occursin(".\n\nNext", out)             # paragraph break preserved
end


@testset "unwrap_lines - extended cases" begin
    raw = "This is hard\nwrapped but not finished?\nYes!\n\nNew paragraph…"
    out = unwrap_lines(raw)

    @test occursin("hard wrapped", out)          # wrap folded
    @test occursin("?\nYes!", out)               # newline kept after '?'
    @test occursin("\n\nNew paragraph…", out)    # paragraph break preserved
end


@testset "Paragraph pipeline - mega-mixed" begin

    raw = """
    Hello\u00A0world!\nThis is a non-ASCII\u200Bline.
    It wraps here,\nwithout ending the sentence.\t

       \t\u200B      \n  \n

    Short.\n
    Another paragraph with emojis 😊😊 and\ttabs.

    最後の段落です。\nIt includes Japanese.
    """

    #split
    paras = split_paragraphs(raw; unwrap = true, normalize = true)
    @test length(paras) == 5                   
    @test startswith(paras[1], "Hello world!")
    @test occursin(r"non-ASCII\s*line\.", PP.strip_zero_width(paras[1]))

    @test paras[3] == "Short."                 # still isolated

    #drop empty
    nonblank = drop_empty_paragraph(paras)
    @test length(nonblank) == 4                  # blank paragraph removed

    # merge short ("Short." merges forward)
    merged = merge_short_paragraphs(nonblank; min_chars = 10)
    @test length(merged) == 3
    @test any(contains.(merged, "Short."))       # merged into first element

    # paragraph windows
    tok(p) = split(p)                            # whitespace tokeniser
    wins = collect(paragraph_windows(merged, 20; tokenizer = tok, stride = 1))
    @test !isempty(wins)                         # iterator works
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


@testset "split_paragraphs - edge cases" begin
    #leading / trailing blank lines collapse
    txt1 = "\n\nAlpha\n\nBeta\n\n\n"
    ps1  = split_paragraphs(txt1; unwrap=false, normalize=false)
    @test ps1 == ["Alpha", "Beta"]

    #paragraph split on blank line containing spaces + tabs
    txt2 = "One.\n  \t \nTwo."
    @test split_paragraphs(txt2)[1] == "One."
    @test split_paragraphs(txt2)[2] == "Two."

    #empty string returns an empty vector
    @test isempty(split_paragraphs(""))

    #windows new-lines (\r\n) treated same as \n
    txt3 = "L1\r\nL2.\r\n\r\nL3."
    @test length(split_paragraphs(txt3)) == 2

    #unwrap=true folds hard-wraps, but leaves genuine paragraph break
    txt4 = "A line\nwrap.\n\nNew para."
    ps4  = split_paragraphs(txt4; unwrap=true)
    @test ps4[1] == "A line wrap."
    @test ps4[2] == "New para."
end


@testset "Paragraph pipeline - chaos corpus" begin
    # helper just for the test
    strip_zws(s) = replace(s, r"[\u200B\u200C\u200D]" => "")

    chaos = """
    \u200BFirst\u00A0paragraph\tstarts here…
    It continues\r\nwithout sentence end.
    Why?\tBecause!

    Some really normal looking text. This text is kinda normal and has some normal sentences and should be a paragraph. It should be a paragraph that is long enough and look normal to most and this test.

    \t\t\n \n      \n  \n   \n     

    第二段落です。\n改行が途中にあります。\r\nでも文章は続きます。😊

    Short.

    مرحبًا بالعالم.\nهذا سطر ثانٍ.\u200B

    Last paragraph ends now!
    """

    # split
    paras = split_paragraphs(chaos; unwrap = true, normalize = true)
    @test length(paras) >= 6
    @test startswith(strip_zws(paras[1]), "First paragraph")
    @test any(startswith.(paras, "Some really normal looking text"))
    @test any(startswith.(paras, "第二段落です。"))
    @test any(startswith.(paras, "مرحبًا بالعالم."))
    @test endswith(strip(paras[end]), "ends now!")

    # drop empty  (verify nothing is blank)
    nonblank = drop_empty_paragraph(paras)
    @test all(!isempty(strip_zws(p)) for p in nonblank)

    # merge short
    merged = merge_short_paragraphs(nonblank; min_chars = 20)
    @test any(contains.(merged, "Short."))          #short, merged forward

    # paragraph windows  (collect by hand)
    tok(p) = split(p)
    win_fn, st = paragraph_windows(merged, 20; tokenizer = tok, stride = 1)
    wins = Vector{Vector{String}}()
    while true
        result = win_fn(st)
        result === nothing && break
        chunk, st = result
        push!(wins, chunk)
    end

    predicate(chunk) = begin
        total = sum(length.(tok.(chunk)))
        total <= 20 || (length(chunk) == 1 && total > 20)
    end

    @test all(predicate.(wins))
end


@testset "paragraph_windows" begin
    dummy_tok(p) = split(p)

    paras = ["alpha beta", "gamma delta epsilon", "zeta"]
    win, st = paragraph_windows(paras, 4; stride = 2, tokenizer = dummy_tok)

    chunks = Vector{Vector{String}}()
    while true
        res = win(st)
        res === nothing && break
        chunk, st = res
        push!(chunks, chunk)
    end

    @test chunks == [["alpha beta"], ["zeta"]]
end


@testset "paragraph_windows - extended coverage" begin
    tok(p) = split(p)                                # toy tokenizer

    gather(win, st) = begin
        out = Vector{Vector{String}}()
        while true
            res = win(st)
            res === nothing && break
            chunk, st = res
            push!(out, chunk)
        end
        out
    end

    #non-overlapping windows (stride = max_tokens)
    paras1 = ["a b", "c d e", "f g h i j", "k"]
    w1, st1 = paragraph_windows(paras1, 5; stride = 5, tokenizer = tok)
    @test gather(w1, st1) == [["a b", "c d e"]]

    #overlapping windows (stride < max_tokens)
    w2, st2 = paragraph_windows(paras1, 5; stride = 1, tokenizer = tok)
    chunks2 = gather(w2, st2)
    @test chunks2[1] == ["a b", "c d e"]
    @test chunks2[2] == ["c d e"]
    @test length(chunks2) == 4

    #first paragraph > cap  -> fallback single-paragraph chunk
    longp = ["word "^50, "short"]
    w3, st3 = paragraph_windows(longp, 10; tokenizer = tok)
    @test gather(w3, st3)[1] == [longp[1]]

    #zero-token paragraph handled
    paras4 = ["", "alpha beta"]
    w4, st4 = paragraph_windows(paras4, 3; tokenizer = tok)
    @test gather(w4, st4) == [["", "alpha beta"]]

    #huge stride skips rest
    w5, st5 = paragraph_windows(paras1, 5; stride = 99, tokenizer = tok)
    @test gather(w5, st5) == [["a b", "c d e"]]

    #max_tokens == 1 edge case
    paras6 = ["a b", "c"]
    w6, st6 = paragraph_windows(paras6, 1; tokenizer = tok)
    @test gather(w6, st6) == [["a b"], ["c"]]

    #empty input
    w7, st7 = paragraph_windows(String[], 4; tokenizer = tok)
    @test gather(w7, st7) == []

    #guard clauses
    @test_throws ArgumentError paragraph_windows(paras1, 0; tokenizer = tok)
    @test_throws ArgumentError paragraph_windows(paras1, 4; stride = 0,
                                                 tokenizer = tok)
end


@testset "Paragraph pipeline - varied-length corpus" begin
    strip_zws(s) = replace(s, r"[\u200B\u200C\u200D]" => "")
    tok(p)       = split(p)       # whitespace tokenizer

    corpus = """
    This is a **long** paragraph containing more than fifty tokens. \
    It rambles on and on about nothing in particular, simply to pad out \
    the length so we can test the long-paragraph fallback inside \
    `paragraph_windows`, ensuring that one paragraph alone may fill— \
    or even overflow—the given token budget.

    Short.

    This is a medium-length paragraph. It has just enough words to sit \
    comfortably between the extremes so we can verify ordinary window \
    packing without falling into the special cases reserved for very \
    small or very large paragraphs.

    Tiny.

    Another exceptionally long paragraph follows to close the corpus. \
    Like the first, it sprawls across many words and clauses, talking \
    at length about absolutely nothing—yet doing so with great gusto— \
    purely in the service of unit-testing. By including this second \
    behemoth we make sure the iterator advances correctly after hitting \
    a paragraph that exceeds the budget.
    """

    # split
    paras = split_paragraphs(corpus; unwrap = true)
    paras = String.(paras) 
    @test length(paras) == 5                   
    @test strip_zws(paras[2]) == "Short."
    @test strip_zws(paras[4]) == "Tiny."

    # merge very short paragraphs forward
    merged = merge_short_paragraphs(paras; min_chars = 10)
    @test length(merged) == 3           # short and tiny merged into neighbours

    # paragraph windows with small budget
    win, st = paragraph_windows(merged, 40; stride = 1, tokenizer = tok)
    chunks = Vector{Vector{String}}()
    while true
        res = win(st)
        res === nothing && break
        chunk, st = res
        push!(chunks, chunk)
    end

    @test length(chunks) == 3      # [Long₁] [Medium] [Long₂]
    @test length(chunks[1]) == 1 && length(tok(chunks[1][1])) > 40
    @test length(chunks[2]) == 1 && length(tok(chunks[2][1])) ≤ 40
    @test length(chunks[3]) == 1 && length(tok(chunks[3][1])) > 40
    @test all(sum(length.(tok.(c))) ≤ 40 || length(c) == 1 for c in chunks)
end


@testset "merge_short_paragraphs" begin
    # helper so tests are explicit
    SS = split_sentences

    #short in the middle merges backward and forward
    pars = String[
        "Tiny.",                                   # 5 chars < 10  -> short
        "This is a medium-length paragraph.",      #> 10 long
        "Another short.",                          #15 chars >= 10  long
        "Here is a very long paragraph that easily exceeds the minimum \
        character requirement, so it should stand alone."
    ]

    merged = merge_short_paragraphs(pars; min_chars = 10)

    @test length(merged) == 3
    @test startswith(merged[1], "Tiny.")              # 'Tiny' merged into next
    @test occursin("medium-length", merged[1])        #medium paragraph present
    @test merged[2] == "Another short."               #untouched
    @test merged[3] == strip(pars[end])               #long para alone

    #first paragraph short, but only one paragraph in doc
    single = ["Short."]
    @test merge_short_paragraphs(single; min_chars = 40) == single

    #last paragraph short merges into previous
    pars2 = ["A real paragraph with content.",
             "End."]
    merged2 = merge_short_paragraphs(pars2; min_chars = 10)
    @test length(merged2) == 1
    @test endswith(merged2[1], "End.")

    #sentence-count criterion
    few_sent = ["One. Two?"]          # 2 sentences
    many_sent = ["One. Two. Three."]  # 3 sentences
    doc = vcat(few_sent, many_sent)

    out = merge_short_paragraphs(doc; min_sents = 3, sentence_splitter = SS)
    @test length(out) == 1            # first merged into second

    #empty input vector
    @test merge_short_paragraphs(String[]; min_chars = 10) == String[]

    #original array untouched
    before = copy(pars)
    _ = merge_short_paragraphs(pars; min_chars = 10)
    @test pars == before
end


@testset "Paragraph pipeline - mixed-scale multi-sentence" begin
    tok(p) = split(p)            # toy tokenizer

    text = """
    This is a **large** paragraph. It contains many sentences—certainly more
    than the average paragraph used in toy examples. We keep talking about
    nothing in particular so that the token count grows. Eventually, this
    paragraph alone will overrun a modest token budget, forcing the window
    iterator to emit it as a single-paragraph chunk.

    Small para.

    This is a medium-length paragraph. It has just enough words to sit
    comfortably between the extremes so we can verify ordinary window
    packing without falling into the special cases reserved for very
    small or very large paragraphs.

    Tiny.

    Another exceptionally large paragraph now follows to close the corpus.
    Like the first, it rambles on and on solely to create a realistic
    distribution of paragraph sizes. Multiple sentences, multiple clauses,
    purely in the service of unit-testing. By including this second
    behemoth we make sure the iterator advances correctly after hitting
    a paragraph that exceeds the budget.
    """

    # split 
    paras = String.(split_paragraphs(text; unwrap = true))
    @test length(paras) == 5
    @test startswith(paras[2], "Small para.")
    @test startswith(paras[4], "Tiny.")

    # merge short 
    merged = merge_short_paragraphs(paras; min_chars = 15)
    @test length(merged) == 3
    @test occursin("Small para.", merged[1])
    @test any(contains.(merged, "Tiny."))

    # window iterator 
    win, st = paragraph_windows(merged, 45; stride = 1, tokenizer = tok)
    chunks = Vector{Vector{String}}()
    while true
        res = win(st)
        res === nothing && break
        chunk, st = res
        push!(chunks, chunk)
    end

    #expectations
    @test length(chunks) == 3                       # [Long] [Med+Short] [Long₂]
    @test length(tok(chunks[1][1])) > 45            # first long para > budget
    @test length(tok(chunks[2][1])) ≤ 45            # merged medium within budget
    @test length(tok(chunks[3][1])) > 45            # second long para > budget
    @test all(sum(length.(tok.(c))) ≤ 45 || length(c) == 1 for c in chunks)
end


@testset "_is_blank_paragraph" begin
    @test _is_blank_paragraph("   \t\u200B")    # blanks + ZWSP
    @test !_is_blank_paragraph("Not blank")
end


@testset "drop_empty_paragraph" begin
    raw = ["Visible paragraph.",
           "   \t",                 # blanks only
           "\u200B\u200C",          # zero-width spaces
           "Another one."]

    cleaned = drop_empty_paragraph(raw)

    @test cleaned == ["Visible paragraph.", "Another one."]  # blanks removed
    @test all(typeof(p) == String for p in cleaned)          # ensured String
end


@testset "filter_paragraphs" begin
    long  = "a"^30
    short = "tiny"
    kept  = filter_paragraphs([long, short]; min_chars = 10)
    @test kept == [long]
end


@testset "filter_paragraphs harder" begin

    paras = [
        "Short.",                                      # 6 chars
        "Emoji 😃😃",                                  # 7 visible chars
        "This paragraph is definitely long enough.",   # 39
        "中文段落。" * "字"^10                          # 12 CJK chars (> 25?)
    ]

    # min_chars = 10 keeps everything except "Short."
    long10 = filter_paragraphs(paras; min_chars = 10)
    @test long10 == paras[2:4]                         # order preserved
    @test all(typeof(p) == String for p in long10)

    # min_chars = 25 keeps only the two genuinely long ones
    long25 = filter_paragraphs(paras; min_chars = 25)
    @test long25 == paras[3:4]

    # Edge: min_chars larger than any paragraph ⇒ empty vector
    @test filter_paragraphs(paras; min_chars = 1000) == String[]
end
