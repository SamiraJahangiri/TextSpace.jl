include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Lemmatization.jl"))



@testset "transform_verb regular cases" begin
    pairs = [
        "tries"     => "try",
        "tried"     => "try",
        "running"   => "run",
        "swimming"  => "swim",
        "stopped"   => "stop",
        "watches"   => "watch",
        "runs"      => "run"
    ]

    for (inflected, lemma) in pairs
        @test transform_verb(inflected) == lemma
    end
end

@testset "lemmatize_word" begin
    # irregular nouns
    @test lemmatize_word("mice")   == "mouse"
    @test lemmatize_word("Knives") == "knife"    # case-insensitive

    # irregular verbs
    @test lemmatize_word("bought") == "buy"
    @test lemmatize_word("ARE")    == "be"       # upper-case input

    # regular verb passes through transform_verb
    @test lemmatize_word("playing") == "play"
end


@testset "lemmatize_text end-to-end" begin
    sentence  = "Mice are running and men bought knives"
    expected  = "mouse be run and man buy knife"
    @test lemmatize_text(sentence) == expected
end


