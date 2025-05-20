include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SentenceProcessing.jl"))



@testset "split_sentences" begin
    txt = "Dr. Smith went to Washington.  It was rainy!  Was it fun?  Yes."
    sents = split_sentences(txt)

    @test length(sents) == 4
    @test sents[1] == "Dr. Smith went to Washington."
    @test sents[2] == "It was rainy!"
    @test sents[end] == "Yes."
end


@testset "strip_outer_quotes" begin
    quoted1 = "\"Hello world!\""
    quoted2 = "“Bonjour tout le monde!”"
    bare1   = strip_outer_quotes(quoted1)
    bare2   = strip_outer_quotes(quoted2)

    @test bare1 == "Hello world!"
    @test bare2 == "Bonjour tout le monde!"
    @test strip_outer_quotes("No quotes.") == "No quotes."
end


@testset "SlidingSentenceWindow" begin
    sents = ["a b c", "d e f", "g h i j k", "l m"]   # lengths: 5,5,9,3 chars
    win   = SlidingSentenceWindow(sents, 12; stride = 2)  # max_tokens≈chars

    collected = collect(win)
    # stride=2 -> first chunk tries 2 sents (len 10) fits, second tries "g h i j k"
    # which is 9 < 12 but because stride=2 it would pick 3rd and 4th (12) -> ok
    @test collected[1] == ["a b c", "d e f"]
    @test collected[2] == ["g h i j k", "l m"]

    # oversize sentence falls back to single-sentence chunk
    long_sent = ["x"^20]   # length 20
    win2 = SlidingSentenceWindow(long_sent, 10)
    @test collect(win2) == [long_sent]                # forced fallback
end
