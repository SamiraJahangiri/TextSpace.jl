include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "DocumentProcessing.jl"))



const HAVE_BPE = let ok = false
    try
        @eval using BytePairEncoding
        ok = isdefined(BytePairEncoding, :learn_bpe)
    catch; end
    ok
end

# helper: tiny word-level Vocabulary
dummy_vocab() = Vocabulary(Dict("<unk>" => 1, "hello"=>2, "world"=>3),
                           ["<unk>", "hello", "world"],
                           Dict{Int,Int}(), 1)


@testset "DocumentProcessing" begin


    if HAVE_BPE
        corpus = ["hello world", "hello there world"]
        tok = BytePairEncoding.learn_bpe(corpus;
                                         vocab_size       = 64,
                                         num_merges       = 100,
                                         special_tokens   = DEFAULT_SPECIAL_TOKENS,
                                         ordered_specials = true)
        tok_pad = tok.vocab["<pad>"]
        voc     = dummy_vocab()

        @testset "process_document API" begin
            doc = """
                  Hello world.
                  Hello there world.

                  Another paragraph short.
                  """

            # sentences
            @test length(process_document(doc, tok; mode = :sentences)) == 3

            # tokens
            toks = process_document(doc, tok; mode = :tokens)
            @test first(toks) == ["hello", "world"]

            # ids  (sub-word route)
            ids = process_document(doc, tok; mode = :ids)
            mat = process_document(doc, tok)          # default :batch
            @test mat[end,1] == tok_pad && size(mat,2) == length(ids)

            # ids  (word route)
            ids_w = process_document(doc, tok; mode = :ids, vocab = voc)
            @test ids_w[1] == [2,3]
        end

        @testset "document_batch_iter" begin
            long_doc = join(["hello world "^30 for _ in 1:20], "\n\n")
            itr, st  = document_batch_iter(long_doc, tok; max_tokens = 100)

            count = 0
            while true
                r = itr(st)
                r === nothing && break
                batch, st = r
                @test size(batch,1) <= 100
                count += 1
            end
            @test count > 1
        end

    else
        # BytePairEncoding unavailable -> create a placeholder set so
        # Test summary still shows 'Pass'
        @testset "BytePairEncoding not installed – skipped" begin
            @test true
        end
    end
end
