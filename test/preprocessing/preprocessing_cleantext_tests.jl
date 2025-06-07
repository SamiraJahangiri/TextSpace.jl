using Test
using Unicode
using Random
using TextSpace.Preprocessing: 
    remove_punctuation,
    normalize_unicode,
    normalize_whitespace,
    remove_emojis,
    remove_accents,
    clean_text,
    strip_zero_width

@testset "remove_punctuation" begin
    txt = "Hello, world! (yes) - test."
    @test remove_punctuation(txt) == "Hello world yes  test"

    txt2 = "100% ©right™ + \$value\$"
    @test remove_punctuation(txt2; remove_symbols = true) ==
          "100 right  value"                 # 'right' stays; symbols removed

    txt3 = "foo\$bar#baz!"
    @test remove_punctuation(txt3; extra_symbols = ['$', '#']) == "foobarbaz"
end


@testset "remove_punctuation (normalised spacing)" begin
    txt = "Hello,   world!   "
    @test remove_punctuation(txt; normalize_whitespace=true) == "Hello world"

    txt2 = "👍🏽🙂 100% + \$"
    @test remove_punctuation(txt2;
           remove_symbols=true, normalize_whitespace=true) == "100"
end


@testset "normalize_unicode" begin
    @test normalize_unicode("e\u0301") == "é"       # NFC default
    @test length(normalize_unicode("é"; form=:NFD)) == 2
    @test_throws ArgumentError normalize_unicode("x"; form=:XYZ)
end


@testset "normalize_whitespace" begin
    s = "  foo \tbar\u200B \r\n baz  "
    @test normalize_whitespace(s; remove_zero_width=true) == "foo bar baz"
    @test normalize_whitespace(s; preserve_newlines=true,
                                   remove_zero_width=true) == "foo bar\n baz"
    @test startswith(normalize_whitespace("  x  "; strip_ends=false), " ")
    @test normalize_whitespace("a\u200B\u200C\u200D"; remove_zero_width=true) == "a"
end


@testset "remove_punctuation - perf sanity" begin
    big = repeat("word!🙂 ", 100_000)    # 1 000 000 bytes
    stripped = remove_punctuation(big; remove_symbols=true)
    stripped = normalize_whitespace(stripped)
    @test length(stripped) == 499_999    # "word " * 100 000 minus last strip
end


_is_punct = c -> occursin(r"\p{P}", string(c))
_is_punct_or_sym = c -> occursin(r"\p{P}|\p{S}", string(c))

@testset "remove_punctuation - curated" begin
    cases = (
        # text, want, remove_symbols
        ("👍🏽🙂‍🌾!", "👍🏽🙂‍🌾", false),
        ("👍🏽🙂‍🌾!", "", true ),

        ("שלום, עולם! Hello?", "שלום עולם Hello", true),

        ("e\u0301clair, café!", "éclair café",  false),

        ("foo\u2060bar\u00A0baz!", "foobar baz", true ),

        ("\uE000\uE001.", "\uE000\uE001",   false),

        ("\u0007Ring!\u000D\r\n", "Ring",   true ),

        ("\$100 ≈ €92.",  "\$100 ≈ €92",   false),
        ("\$100 ≈ €92.",  "100 92",    true ),
    )

    for (txt, want, rs) in cases
        got = remove_punctuation(txt; remove_symbols = rs)
        got  = normalize_unicode(got)
        got  = normalize_whitespace(got; strip_ends=true,
                                          remove_zero_width=true)
        want = normalize_unicode(want)
        want = normalize_whitespace(want; strip_ends=true,
                                          remove_zero_width=true)
        @test got == want

        if rs                   # we stripped symbols as well
            @test !any(_is_punct_or_sym, got)
        else                    # only punctuation stripped
            @test !any(_is_punct, got)
        end
    end
end


@testset "remove_emojis" begin
    msg  = "I ❤️ pizza 🍕 and burgers 🚀!"
    cleaned = remove_emojis(msg)
    @test !occursin(r"\p{So}", cleaned)      # no symbols-emoji category left
end


@testset "remove_accents" begin
    accented = "Café naïve fiancé déjà vu"
    @test remove_accents(accented) == "Cafe naive fiance deja vu"
end


@testset "remove_emojis - curated" begin
    #base cases
    @test remove_emojis("") == ""
    @test remove_emojis("plain ASCII") == "plain ASCII"

    msg = "I ❤️ pizza 🍕 and burgers 🚀!"
    @test remove_emojis(msg) == "I  pizza  and burgers !"  # hearts & pizza & rocket removed

    #ZWJ family + skin-tone
    fam = "👨‍👩‍👧‍👦👩🏽‍🚀"
    clean = remove_emojis(fam; keep_zero_width = true)

    @test all(c == '\u200D' for c in clean) #all chars are ZWJ
    @test length(clean) == 4 # exactly four of them

    #flags
    flag = "Go 🇨🇦!"
    @test remove_emojis(flag) == "Go !"
end


@testset "remove_emojis - random hammer" begin
    rng       = MersenneTwister(2025)
    #400 random emoji code-points
    emoji_pool = [Char(cp) for cp in vcat(0x1F300:0x1F5FF,
                                          0x1F600:0x1F64F,
                                          0x1F680:0x1F6FF,
                                          0x1F900:0x1F9FF,
                                          0x1FA70:0x1FAFF)]
    for _ in 1:500
        s = String(rand(rng, emoji_pool, rand(rng, 1:8))) * " text "
        cleaned = remove_emojis(s)
        @test cleaned == " text "
    end
end


@testset "remove_emojis - performance & idempotence" begin
    big = repeat("word🙂 ", 200_000)                #about 1.2 MB
    @time cleaned = remove_emojis(big)
    @test cleaned == repeat("word ", 200_000)       #emoji all gone
    @test cleaned == remove_emojis(cleaned)         
end


@testset "remove_accents – curated" begin
    #basic Latin with common diacritics
    s = "Café naïve fiancé déjà vu"
    @test remove_accents(s) == "Cafe naive fiance deja vu"

    #pre-composed vs. decomposed code-points
    composed   = "Élève"
    decomposed = Unicode.normalize(composed, :NFD)
    @test remove_accents(composed)   == "Eleve"
    @test remove_accents(decomposed) == "Eleve"   # idempotent

    #non-Latin scripts with combining marks (even Greek tonos)
    greek = "Παράδειγμα"           # Παράδειγμα → Παραδειγμα
    @test remove_accents(greek) == "Παραδειγμα"

    #scripts without diacritics are unchanged
    chinese = "汉字漢字"
    @test remove_accents(chinese) == chinese

    #ligature ﬂ remains (we're only stripping Mn, not doing NFKD)
    @test remove_accents("ﬂ") == "ﬂ"
end


@testset "remove_accents - random fuzz" begin
    rng = MersenneTwister(2025)
    #build an alphabet of base letters + common combining marks
    bases  = [Char(c) for c in 0x0061:0x007A] #a-z
    marks  = [Char(c) for c in 0x0300:0x036F] #combining Diacriticals
    alphabet = vcat(bases, marks)

    for _ in 1:500
        len = rand(rng, 1:40)
        s   = String(rand(rng, alphabet, len))
        out = remove_accents(s)

        #should contain no Mn marks
        @test !occursin(r"\p{Mn}", out)

        @test remove_accents(out) == out
    end
end


@testset "remove_accents - performance" begin
    long = repeat("àéîöû ", 200_000)      #around 1.2 MB
    @time cleaned = remove_accents(long)
    @test cleaned == repeat("aeiou ", 200_000)
end


@testset "clean_text end-to-end" begin
    raw     = "  Café✨ isn't bad!  😀  "
    expect  = "cafe isnt bad"

    out = clean_text(raw;
                     do_remove_accents     = true,
                     do_remove_punctuation = true,
                     do_remove_emojis      = true,
                     case_transform        = :lower)

    @test out == expect
end


@testset "clean_text - end-to-end (legacy example)" begin
    raw    = "  Café✨ isn't bad!  😀  "
    expect = "cafe isnt bad"

    out = clean_text(raw;
                     do_remove_accents     = true,
                     do_remove_punctuation = true,
                     do_remove_emojis      = true,
                     case_transform        = :lower)

    @test out == expect
end


@testset "clean_text - individual flags" begin
    # 1. keep punctuation, just remove accents
    @test clean_text("Élan, déjà vu";
                     do_remove_accents = true,
                     case_transform    = :none) ==
          "Elan, deja vu"

    # 2. uppercase, keep accents
    @test clean_text("élève"; case_transform = :upper) == "ÉLÈVE"

    # 3. remove Unicode symbols too (€, +)
    @test clean_text("price €5 + tax";
                     do_remove_symbols = true) == "price 5 tax"

    # 4. extra_symbols list
    @test clean_text("a\$b\$c"; extra_symbols = ['\$']) == "abc"
end


@testset "clean_text - idempotence & error branch" begin
    s   = "Núñez🚴‍♂️"
    out = clean_text(s;
                     do_remove_accents = true,
                     do_remove_emojis  = true,
                     case_transform    = :lower)
    @test out == clean_text(out)                       # idempotent
    @test_throws ArgumentError clean_text("x"; case_transform = :camel)
end



@testset "clean_text - long multi-paragraph weirdness" begin
    raw = """
    Café\u00A0— élève\u202F— résumé!\u200B 🚀🎉👍🏽

    hello world!

    ‎\u200Efoo\u200Fbar\u2060baz\u2062

    *(@%#\$),,,...

    Price:\u00A0€5,00 + tax \$ (≈ ₅₀%) १२३४५६
    👨‍👩‍👧‍👦 flag: 🇨🇦  Zalgo: Z̴͚͋͗ͅa̶͙͖͋̍ḹ̸̛͇g̸̩͐̈́ỏ̵̟!
    """

    out = clean_text(raw;
                     unicode_normalize      = true,
                     do_remove_accents      = true,
                     do_remove_punctuation  = true,
                     do_remove_symbols      = true,
                     do_remove_emojis       = true,
                     case_transform         = :lower,
                     extra_symbols          = ['$', '#'])

    #no punctuation / symbols / combining marks
    @test !occursin(r"\p{P}",  out)
    @test !occursin(r"\p{S}",  out)
    @test !occursin(r"\p{Mn}", out)

    #no code-points in U+1F300-U+1FAFF  (emoji range)
    @test all(!(0x1F300 <= UInt32(c) <= 0x1FAFF) for c in out)

    #no zero-width characters
    @test !occursin(r"[\u200B-\u200F\u2060-\u206F]", out)

    #all letters are lowercase
    @test all(!isletter(c) || islowercase(c) for c in out)

    #whitespace: single spaces, no newlines
    @test !occursin('\n', out)
    @test !occursin(r"\s{2,}", out)

    #idempotence
    @test out == clean_text(out;
                            unicode_normalize      = true,
                            do_remove_accents      = true,
                            do_remove_punctuation  = true,
                            do_remove_symbols      = true,
                            do_remove_emojis       = true,
                            case_transform         = :lower,
                            extra_symbols          = ['$', '#'])
end


@testset "strip_zero_width" begin
    s = "a\u200Bb\u200Dc"
    @test strip_zero_width(s) == "abc"

    nl = "α\u200B\n\u200Bβ"
    @test strip_zero_width(nl) == "α\nβ"         # keeps newline

    empty = "\u200B\u200C"
    @test strip_zero_width(empty) == ""          # all removed
end
