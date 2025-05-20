#=
# Common and Useful Embedding Types in NLP and LLMs

## 1. Text-Based Embeddings
- **Character Embeddings** - Capture morphological patterns (e.g., CharLSTM, FastText).
- **Subword Embeddings** - Handle rare words using BPE, SentencePiece (e.g., WordPiece, Unigram).
- **Word Embeddings** - Fixed-length word vectors (e.g., Word2Vec, GloVe, FastText).
- **Phrase Embeddings** - Represent multi-word expressions (e.g., "New York", "deep learning").
- **Sentence Embeddings** - Capture sentence-level meaning (e.g., SBERT, Universal Sentence Encoder).
- **Paragraph Embeddings** - Represent longer text spans (e.g., Doc2Vec).
- **Document Embeddings** - Whole document representation (aggregated or learned).

## 2. Model-Specific Embeddings in LLMs
- **Token Embeddings** - Core subword embeddings used by transformers.
- **Positional Embeddings** - Encode token order (Absolute, Relative, or RoPE).
- **Contextualized Embeddings** - Dynamic word representations (e.g., BERT, GPT).
- **Segment Embeddings** - Distinguish sentence pairs in models like BERT.
- **Task-Specific Embeddings** - Used in instruction-tuned models (e.g., FLAN-T5, ChatGPT).
=#

module TextSpace

using Unicode
using Reexport

#load before using 
include(joinpath(@__DIR__, "preprocessing", "Preprocessing.jl"))
#now use
@reexport using .Preprocessing

# high-level embeddings
include(joinpath(@__DIR__, "embeddings", "CharacterEmbeddings.jl"))
@reexport using .CharacterEmbeddings 

include(joinpath(@__DIR__, "embeddings", "WordEmbeddings.jl"))
@reexport using .WordEmbeddings 

# package‑wide utilities
# include("utils/StringExtras.jl")
end #END MODULE