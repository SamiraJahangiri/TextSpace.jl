using Test
using Unicode
using TextSpace.Preprocessing:
    normalize_unicode,
    normalize_whitespace


@testset "normalize_unicode" begin
    decomposed = "Cafe\u0301"                 # "Café" (e + COMBINING ACUTE)
    composed   = "Café"                       # NFC form

    @test normalize_unicode(decomposed) == composed          # default :NFC
    @test normalize_unicode(composed; form = :NFD) != composed   # round‑trip differs
    @test Unicode.normalize(composed, :NFD) ==
          normalize_unicode(composed; form = :NFD)            # same API
end


@testset "normalize_unicode - UTF-8 hammer" begin
    #ASCII and empty string are no-ops    
    @test normalize_unicode("Plain ASCII!") == "Plain ASCII!"
    @test normalize_unicode("") == ""

    #ligature split under NFKD / kept under NFC 
    lig = "ﬁ"                                  # U+FB01
    @test normalize_unicode(lig) == lig        # NFC keeps it
    @test normalize_unicode(lig; form = :NFKD) == "fi"   # compatibility split

    #canonical re-ordering + composition    
    weird  = "a\u0303\u0301"                   # a + ˜ + ´ (tilde first)
    canon  = normalize_unicode(weird)          # NFC result
    @test canon == "ã́"                        # composed & reordered
    @test normalize_unicode(canon) == canon    # idempotent

    #ZWJ emoji & family               
    fam = "👨‍👩‍👧‍👦"
    @test normalize_unicode(fam) == fam
    @test normalize_unicode(fam; form = :NFD) == fam

    #Arabic presentation form collapses under NFKC   
    arab_pf = "\uFB50"                          # ALEF WASLA isolated form
    @test normalize_unicode(arab_pf; form = :NFKC) == "ٱ"  # U+0671

    #unsupported form raises (only when guard present) 
    guard_present = try
        normalize_unicode("x"; form = :XYZ); false
    catch e
        isa(e, ArgumentError)
    end
    guard_present && @test_throws ArgumentError normalize_unicode("abc"; form = :XYZ)
end


@testset "normalize_whitespace — defaults" begin
    txt = "  foo\t\tbar\nbaz   "
    @test normalize_whitespace(txt) == "foo bar baz"          # collapsed + trimmed
end


@testset "preserve_newlines = true" begin
    txt  = "foo \t  bar\nbaz\r\nqux"
    out  = normalize_whitespace(txt; preserve_newlines = true)

    @test out == "foo bar\nbaz\nqux"                          # spaces collapsed, NL kept
    @test occursin('\n', out)
    @test !occursin('\t', out)                                # tabs gone
end


@testset "strip_ends = false" begin
    txt = "   foo bar   "
    @test normalize_whitespace(txt; strip_ends = false) ==
          " foo bar "                                         # leading/trailing kept
end


@testset "remove_zero_width = true" begin
    zwsp = "\u200B"                               # zero-width space
    txt  = "foo" * zwsp * "bar  baz"
    @test normalize_whitespace(txt; remove_zero_width = true) ==
          "foobar baz"                            # zwsp removed, blanks collapsed
end


@testset "normalize_whitespace - hammer" begin
    zwsp = "\u200B"
    zwnj = "\u200C"
    nbsp = "\u00A0"
    crlf = "Line1 \r\n   Line2"
    many = "  A  \t\tB\n\nC  "

    #basic collapse (no flags)
    @test normalize_whitespace("foo   bar\tbaz") == "foo bar baz"

    #strip_ends = false
    s = "   foo   "
    @test normalize_whitespace(s; strip_ends = false) == " foo "

    #preserve_newlines = true
    nl = normalize_whitespace(crlf; preserve_newlines = true)
    @test nl == "Line1\n Line2"  # single space kept before LF

    #remove_zero_width = true
    zw = normalize_whitespace("a$(zwsp)b$(zwnj)c";
                              remove_zero_width = true)
    @test zw == "abc"  # zero-width chars removed, no spaces inserted

    #all three flags together
    txt  = "$(nbsp)foo$(zwsp)\tbar  "
    comb = normalize_whitespace(txt;
                                remove_zero_width = true,
                                strip_ends        = true,
                                preserve_newlines = false)
    @test comb == "foo bar"

    #multi-LF retained when preserve_newlines = true
    dd = normalize_whitespace(many; preserve_newlines = true)
    @test dd == "A B\n\nC"

    #empty string fast-path
    @test normalize_whitespace("") == ""

    #already normalised stays unchanged
    @test normalize_whitespace("foo bar") == "foo bar"
end


