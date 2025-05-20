include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Stemming.jl"))


@testset "porter_stem basic cases" begin
    pairs = [
        "caresses"    => "caress",
        "ponies"      => "poni",
        "running"     => "run",
        "relational"  => "relat",
        "conditional" => "condit",
        "meeting"     => "meet",
        "cats"        => "cat",
        "happy"       => "happi"
    ]

    for (word, stem) in pairs
        @test porter_stem(word) == stem
    end
end


@testset "stem_text end-to-end" begin
    sentence = "caresses ponies running cats"
    expected = "caress poni run cat"
    @test stem_text(sentence) == expected
end


