using Test
using Unicode
using TextSpace.Preprocessing: tokenize_char, char_tokens


@testset "tokenize_char" begin
    txt = "Café 😊"

    # default: NFC, no spaces, original case
    toks = tokenize_char(txt)
    @test toks == ["C","a","f","é","😊"]

    # lower-case, keep space
    toks2 = tokenize_char(txt; lower = true, keep_space = true)
    @test toks2 == ["c","a","f","é"," ","😊"]

    # combined accents survive when skipping normalisation (:NFD)
    nfd = Unicode.normalize("é", :NFD)          # 'e' + combining acute
    toks3 = tokenize_char(nfd; normalize = false)
    @test length(toks3) == 1                    # one grapheme cluster
end


@testset "tokenize_char - complex graphemes" begin
    # ZWJ family      👩🏽‍🚀  (woman + skin-tone + zwj + rocket)
    @test tokenize_char("👩🏽‍🚀") == ["👩🏽‍🚀"]

    # Flag sequence   🇨🇦  (regional indicators)
    @test tokenize_char("🇨🇦") == ["🇨🇦"]

    # Hindi combining matra  क + ि  to कि  (single grapheme)
    dev = "कि"
    @test tokenize_char(dev) == ["कि"]
end


@testset "tokenize_char - whitespace handling" begin
    s = "a \t\n b"
    @test tokenize_char(s; keep_space=false) == ["a","b"]
    toks = tokenize_char(s; keep_space=true)
    @test count(t -> isspace(first(t)), toks) == 4
end


@testset "tokenize_char - lower & NFC idempotence" begin
    src  = "Núñez"
    toks = tokenize_char(src; lower=true)
    @test join(toks) == lowercase(Unicode.normalize(src, :NFC))
end


@testset "tokenize_char - error branch" begin
    @test_throws ArgumentError tokenize_char("x"; form=:XYZ)
end


@testset "char_tokens" begin
    #ASCII, no eos
    @test char_tokens("hello") == ["h","e","l","l","o"]

    #unicode grapheme
    @test char_tokens("café") == ["c","a","f","é"]
    @test char_tokens("👩🏽‍🚀") == ["👩🏽‍🚀"]     # single grapheme

    #end-of-word marker
    @test char_tokens("hi"; eos="</w>") == ["h","i","</w>"]

    #empty string edge-case
    @test char_tokens("") == String[]

    substr = SubString("hello-world", 1, 5)   # "hello"
    @test char_tokens(substr) == ["h","e","l","l","o"]
end


@testset "char_tokens - robust Unicode hammer" begin
    using Unicode

    zwsp   = "\u200B"                       # ZERO-WIDTH SPACE
    flagUS = "🇺🇸"                          # two regional indicators, one grapheme
    family = "👨‍👩‍👧‍👦"                   # family emoji (ZWJ sequence)
    astro  = "👩🏽‍🚀"                       # skin-tone + ZWJ
    kiss   = "👩‍❤️‍💋‍👨"                  # complex ZWJ chain
    combé  = "e\u0301"                      # e + COMBINING ACUTE
    multiC = "a\u0300\u0316"                # a + grave + combining sub-dot
    loneCM = "\u0301"                       # COMBINING ACUTE by itself
    empty  = ""

    words  = [ "hello",
               flagUS, family, astro, kiss,
               combé, multiC, loneCM,
               "abc$zwsp",                   # embedded zero-width space
               empty ]

    # helper to build the reference using Unicode.graphemes (ground truth)
    expected(w; eos=nothing) = begin
        toks = [String(g) for g in Unicode.graphemes(String(w))]
        eos === nothing || push!(toks, eos)
        toks
    end

    #for every quirky word, result matches Unicode.graphemes
    for w in words
        @test char_tokens(w) == expected(w)
        @test char_tokens(w; eos="</w>") == expected(w; eos="</w>")
    end

    #every output token is a non-empty, valid UTF-8 String
    for w in words, tok in char_tokens(w)
        @test isa(tok, String) && !isempty(tok) && isvalid(tok)
    end

    #`eos` marker appended exactly once           
    sample = char_tokens("hi"; eos="</e>")
    @test sample[end] == "</e>" && count(==("</e>"), sample) == 1

    #SubString input behaves identically       
    sub = SubString("😊-xyz", 1, 1)            # the smiley face only
    @test char_tokens(sub) == [ "😊" ]
end


@testset "char_tokens - paragraph smoke test" begin
    para = "The café—naïve coöperation 🤯👩🏽‍🚀.\nLine2 with  汉字  and Ωmega."

    tokens = char_tokens(para)                   # yes, this keeps spaces
    @test length(tokens) == length(Unicode.graphemes(para))
    @test all(isvalid, tokens) && !any(==(""), tokens)
end
