include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "TextVectorization.jl"))




@testset "pad_sequences" begin
    seqs = [[1,2,3,4], [5,6]]

    m = pad_sequences(seqs; pad_value = 0)          # default maxlen = 4
    @test size(m) == (4,2)
    @test m[:,1] == [1,2,3,4]
    @test all(m[3:4,2] .== 0)                      # padded

    # explicit maxlen > longest  ⇒ extra pad
    m2 = pad_sequences(seqs; maxlen = 5, pad_value = -1)
    @test m2[5,1] == -1
    @test m2[3:5,2] == [-1,-1,-1]

    # truncation, keep tail when trunc = :pre
    m3 = pad_sequences(seqs; maxlen = 3, trunc = :pre)
    @test m3[:,1] == [2,3,4]                       # tail kept
end


@testset "one_hot" begin
    oh = one_hot([2,4,1], 5)
    @test oh[2,1]            # id 2 hot in column 1
    @test oh[4,2]            # id 4 hot in column 2
    @test oh[1,3]            # id 1 hot in column 3
    @test count(>(0), oh) == 3  # exactly three ones
end


@testset "BoW helpers" begin
    seq  = [1,2,2,5]
    vsize = 5
    @test bow_counts(seq, vsize) == [1, 2, 0, 0, 1]   # ← fixed

    docs = [[1,2], [2,2,4]]
    bm   = bow_matrix(docs, vsize)
    @test bm[:,1] == [1,1,0,0,0]
    @test bm[:,2] == [0,2,0,1,0]
end


@testset "tfidf_matrix (tiny example)" begin
    docs = [[1,1], [1,2]]
    tfidf = tfidf_matrix(docs, 2; smooth_idf = 0)   # raw IDF

    @test isapprox(tfidf[1,1], 0.0; atol=1e-12)     # idf=0 because word 1 in both docs
    @test tfidf[2,2] > 0.0                          # idf(log 2) > 0 for word 2
end


@testset "batch_iter" begin
    seqs = [[1,2,3,4], [5,6], [7]]
    itr, state = batch_iter(seqs, 2; shuffle = false, pad_value = 0)

    (mat1, sub1), state = itr(state)
    @test sub1 == [1,2]
    @test size(mat1) == (4, 2)              # longest seq in batch = 4

    (mat2, sub2), state = itr(state)
    @test sub2 == [3]                       # last doc
    @test size(mat2) == (1, 1)              # longest seq in this batch = 1
    @test mat2[1,1] == 7                    # value preserved, no padding needed

    @test itr(state) === nothing            # iterator exhausted
end

