using TextSpace
using Test

using Random
using Downloads 

@testset "TextSpace.jl Test Suite" begin
    # Preprocessing Tests
    @testset "Preprocessing" begin
        include("preprocessing/preprocessing_cleantext_tests.jl")
        include("preprocessing/preprocessing_textnormalization_tests.jl")
        include("preprocessing/preprocessing_tokenization_tests.jl")
        include("preprocessing/preprocessing_char_tests.jl")
        include("preprocessing/preprocessing_sentence_tests.jl")
        include("preprocessing/preprocessing_paragraph_tests.jl")
        include("preprocessing/preprocessing_subword_pipeline_tests.jl")
    end

    # Pipeline Tests
    @testset "Pipelines" begin
        include("pipeline/preprocessing_pipeline_tests.jl")
    end

    # Basic tests
    @testset "Basic Tests" begin
        # Smoke test
        @test true
        
        # Root test
        text1 = "Hello, World!"
        @test text1 == "Hello, World!"
    end

    # You can uncomment and organize these when ready
    # @testset "Embedding Tests" begin
    #     include("SubwordEmbeddings/subword_embeddings_test_gateway.jl")
    #     include("WordEmbeddings/word_embeddings_test_gateway.jl")
    #     include("CharacterEmbeddings/character_embeddings_test_gateway.jl")
    # end
    
    # @testset "Utility Tests" begin
    #     include("util-tests/__init__.jl")
    # end
end
