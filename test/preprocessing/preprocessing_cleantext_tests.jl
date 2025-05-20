include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "CleanText.jl"))


@testset "remove_punctuation" begin
    txt = "Hello, world! (yes) - test."
    @test remove_punctuation(txt) == "Hello world yes  test"

    txt2 = "100% ©right™ + \$value\$"
    @test remove_punctuation(txt2; remove_symbols = true) ==
          "100 right  value"                 # ← “right” stays; symbols removed

    txt3 = "foo\$bar#baz!"
    @test remove_punctuation(txt3; extra_symbols = ['$', '#']) == "foobarbaz"
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



