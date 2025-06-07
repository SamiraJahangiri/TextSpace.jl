
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Tokenization.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "CharProcessing.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "TextNormalization.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SentenceProcessing.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "ParagraphProcessing.jl"))



#test paragraph processing
include("preprocessing_paragraph_tests.jl")


#test sentence processing
include("preprocessing_sentence_tests.jl")


#test char preprocessing
include("preprocessing_char_tests.jl")


#test the text tokenization
include("preprocessing_tokenization_tests.jl")


#test the textnormalization
include("preprocessing_textnormalization_tests.jl")


#test clean text
include("preprocessing_cleantext_tests.jl")







