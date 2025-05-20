
# include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Vocabulary.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Tokenization.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "CharProcessing.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "TextNormalization.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "TextVectorization.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SubwordTokenization.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SentenceProcessing.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "ParagraphProcessing.jl"))



#test char pipeline preprocessing
include("preprocessing_char_pipeline_tests.jl")

#test char preprocessing
include("preprocessing_char_tests.jl")

#test document processing
include("preprocessing_document_tests.jl")

#test paragraph processing
include("preprocessing_paragraph_tests.jl")

#test sentence processing
include("preprocessing_sentence_tests.jl")

#test subword tokenization
include("preprocessing_subwordtokenization_tests.jl")

#test clean text
include("preprocessing_cleantext_tests.jl")

#test stemming
include("preprocessing_lemmatization_tests.jl")

#test stemming
include("preprocessing_stemming_tests.jl")

#test the textnormalization
include("preprocessing_textnormalization_tests.jl")

#test the vocabulary
include("preprocessing_vocaculary_tests.jl")

#test the text vectorization
include("preprocessing_textvectorization_tests.jl")

#test the text tokenization
include("preprocessing_tokenization_tests.jl")
