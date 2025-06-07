using Test
using TextSpace.Preprocessing:
    basic_tokenize,
    strip_punctuation,
    ngrams,
    tokenize,
    tokenize_batch

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


@testset "basic_tokenize - UTF-8 hammer" begin
    zwsp   = "\u200B";  nbsp  = "\u00A0"
    rle    = "\u202B";  pdf   = "\u202C"
    combé  = "e\u0301"; ligfi = "ﬁ"
    flag   = "🇯🇵";      emoji = "👩🏽‍🚀";  fam = "👨‍👩‍👧‍👦"

    paragraph = """
    Line1: 123_456 can't – $(flag) wow!$zwsp ..

    ... !!!

    Line2 tabs\tand  $nbsp non-breakers.  $(emoji),$(fam)
    Line3 $(rle)مرحبا$pdf CJK=漢字; Greek=Ωmega; Lig=$ligfi; Comb=$combé
    """

    #keep_whitespace = false 
    toks = basic_tokenize(paragraph)

    @test "can't" in toks
    @test "漢字"  in toks
    @test "مرحبا" in toks

    # at least one non-alnum token that is emoji / symbol survived
    @test any(t -> !occursin(r"^[\p{L}\p{N}_']+$", t), toks)

    # punctuation isolated
    @test ("," in toks) && ("!" in toks) && (";" in toks)

    # no whitespace tokens
    @test !any(t -> occursin(WHITESPACE_REGEX, t), toks)

    #keep_whitespace = true
    toks_ws = basic_tokenize(paragraph; keep_whitespace=true)

    @test any(t -> occursin(WHITESPACE_REGEX, t), toks_ws)   # pure spaces
    @test any(t -> t == "\n" || t == "\r\n", toks_ws)        # newlines
    @test length(toks_ws) > length(toks)

    #loss-less round-trip (ignoring ws collapse) 
    rebuilt = join(toks_ws, "")
    @test replace(rebuilt, r"\s+" => " ") == replace(paragraph, r"\s+" => " ")
end


@testset "strip_punctuation" begin
    #plain word unchanged
    @test strip_punctuation("hello") == "hello"

    #leading and trailing ASCII punctuation
    @test strip_punctuation("(hello)!") == "hello"

    #quotation marks & apostrophes (curly and straight)
    @test strip_punctuation("“quoted”") == "quoted"
    @test strip_punctuation("'single'") == "single"
    @test strip_punctuation("’smart’")  == "smart"

    #mixed: punctuation on one side only
    @test strip_punctuation("world?")  == "world"
    @test strip_punctuation("[array")  == "array"

    #token that is *nothing but* punctuation -> empty string
    @test strip_punctuation("!!!") == ""

    #unicode punctuation *inside* word remains
    @test strip_punctuation("rock'n'roll") == "rock'n'roll"

    #emojis & symbols untouched
    @test strip_punctuation("🚀rocket🚀") == "🚀rocket🚀"  # leading char not in set

    #SubString input behaves identically
    s = ">>>abc<<<"
    sub = SubString(s, 4, 6)           # "abc"
    @test strip_punctuation(sub) == "abc"
end


@testset "ngrams" begin
    #unigrams are just the input
    toks = ["a","b","c"]
    @test ngrams(toks, 1) === toks

    #bigrams
    @test ngrams(toks, 2) == ["a_b", "b_c"]

    #trigrams
    @test ngrams(toks, 3) == ["a_b_c"]

    #n larger than length -> empty
    @test isempty(ngrams(toks, 4))

    #unicode tokens
    uni = ["漢字", "👩🏽‍🚀", "Ωmega"]
    @test ngrams(uni, 2) == ["漢字_👩🏽‍🚀", "👩🏽‍🚀_Ωmega"]

    #empty input
    @test isempty(ngrams(String[], 2))

    #n = 0 or <0 throws ArgumentError (only if you applied the refactor)
    @test_throws ArgumentError ngrams(toks, 0)
    @test_throws ArgumentError ngrams(toks, -1)
end


@testset "ngrams - UTF-8 hammer" begin
    #hand-crafted token list with emoji, skin-tone, CJK, underscores 
    toks = ["👩🏽‍🚀",     # skin-tone ZWJ emoji
            "🤯",
            "漢字",        # CJK
            "Ωmega",       # Greek letter
            "a_b",         # already contains an underscore
            "👍"]

    # bigrams
    @test ngrams(toks, 2) ==
          ["👩🏽‍🚀_🤯", "🤯_漢字", "漢字_Ωmega",
           "Ωmega_a_b", "a_b_👍"]

    # trigrams
    @test ngrams(toks, 3) ==
          ["👩🏽‍🚀_🤯_漢字", "🤯_漢字_Ωmega",
           "漢字_Ωmega_a_b", "Ωmega_a_b_👍"]

    # n larger than length ⇒ empty Vector
    @test isempty(ngrams(toks, 10))

    # original token vector must be unchanged (no mutation)
    @test toks == ["👩🏽‍🚀", "🤯", "漢字", "Ωmega", "a_b", "👍"]

    #paragraph-sourced tokens: length & boundary sanity    
    paragraph = """
    Hello 👩🏽‍🚀!  漢字 and Ωmega –
    
    \t \n smile 😊.  
    """

    # tokenise without stripping punctuation so we get a varied list
    t2 = basic_tokenize(paragraph; keep_whitespace=false) # e.g. ["Hello","👩🏽‍🚀","!",…]

    n  = 2
    ng = ngrams(t2, n)

    # length formula |t| - n + 1
    @test length(ng) == length(t2) - n + 1

    # first & last n-grams match manual joins
    @test first(ng) == join(t2[1:2], '_')
    @test last(ng)  == join(t2[end-1:end], '_')

    #edge cases: empty input and invalid n   
    @test isempty(ngrams(String[], 3))

    # Only include these if you adopted the ArgumentError refactor:
    @test_throws ArgumentError ngrams(toks, 0)
    @test_throws ArgumentError ngrams(toks, -2)
end


@testset "tokenize - end-to-end" begin
    # simple default pipeline 
    @test tokenize("Hello, World!") == ["hello", "world"]

    #stop-word removal
    @test tokenize("This is a test.";
                   remove_stopwords=true) == ["this", "test"]

    #keep punctuation & case
    @test tokenize("Wow!?"; strip_punctuation=false, lower=false) ==
          ["Wow", "!", "?"]

    # whitespace retention introduces space & newline tokens
    tws = tokenize("Hi \nthere";
                   keep_whitespace=true,
                   strip_punctuation=false,
                   lower=false)
    @test any(t -> t == "\n" || occursin(r"^\s+$", t), tws)

    #unicode survives
    utoks = tokenize("漢字 👩🏽‍🚀")
    @test "漢字" in utoks
    @test any(t -> occursin(r"[👩🏽🚀]", t), utoks)

    #n-grams
    tri = tokenize("a b c d";
                   ngram=3, strip_punctuation=false, lower=false)
    @test tri == ["a_b_c", "b_c_d"]

    #internal apostrophes kept while outer punctuation removed
    @test tokenize("(rock'n'roll)") == ["rock'n'roll"]

    #all-punctuation token pruned to empty -> disappears
    @test tokenize("!!!") == String[]
end


@testset "tokenize_batch" begin
    corpus = ["Hello, World!",
              "漢字 👩🏽‍🚀",
              "This   is\na test."]

    #default (single-thread) path
    batch = tokenize_batch(corpus)
    @test length(batch) == 3
    @test batch[1] == ["hello", "world"]          # inherited defaults
    @test "漢字" in batch[2]                       # Unicode survives

    #keyword forwarding works
    no_sw = tokenize_batch(corpus;
                           strip_punctuation=false,
                           lower=false,
                           remove_stopwords=true)
    @test !("is" in no_sw[3])                          # stop-word removed
    @test no_sw[1] == ["Hello", ",", "World", "!"]

    #empty input -> empty output
    @test isempty(tokenize_batch(String[]))

    #threaded path returns same result (if multiple threads)
    if Threads.nthreads() > 1
        thr = tokenize_batch(corpus; threaded=true)
        @test thr == batch
    end
end


