include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Preprocessing.jl"))



@testset "preprocess_for_char_embeddings" begin
    
    #raw string - new vocabulary
    
    txt  = "Hello 😊"
    out  = preprocess_for_char_embeddings(txt)          # build vocab

    #cleaning left the visible text unchanged
    @test out.cleaned_text == txt

    #tokenisation: default keeps case, drops spaces
    @test out.chars == ["H","e","l","l","o","😊"]

    # <unk> exists and has a valid positive id
    @test haskey(out.vocabulary.token2id, "<unk>")
    @test out.vocabulary.unk_id ≥ 1

    #char_ids are a 1-to-1 mapping of the returned characters
    @test out.char_ids ==
          [out.vocabulary.token2id[c] for c in out.chars]

    
    #re-using an existing vocabulary
    txt2 = "Hola"
    out2 = preprocess_for_char_embeddings(txt2; vocab = out.vocabulary)

    #should not create a brand-new Vocabulary object
    @test out2.vocabulary === out.vocabulary

    #all ids are within the known range
    @test all(1 ≤ id ≤ length(out.vocabulary.id2token) for id in out2.char_ids)

    
    #file-path input & unknown characters fall back to <unk>
    mktemp() do path, io          # ← path first, stream second
        write(io, "¿Qué?")
        close(io)
    
        out3 = preprocess_for_char_embeddings(path; vocab = out.vocabulary)
    
        # first character should be mapped to the unk id
        @test out3.char_ids[1] == out.vocabulary.unk_id
    end
end



@testset "preprocess_for_char_embeddings - additional coverage" begin
    
    #clean_options + char_options flags
    txt = "Go :)"
    res = preprocess_for_char_embeddings(txt;
                                        clean_options = Dict(
                                            :case_transform        => :lower,
                                            :do_remove_punctuation => true
                                        ),
                                        char_options  = Dict(:keep_space => true))

    @test res.cleaned_text == "go"               # no trailing blank after normalisation
    @test res.chars == ["g","o"]
    @test res.char_ids[1:2] ==
        [res.vocabulary.token2id[c] for c in res.chars]     # length 2



    # add_new = true really grows the vocabulary and updates counts
    base = preprocess_for_char_embeddings("abc")    # fresh vocab
    orig_vocab = base.vocabulary
    orig_len   = length(orig_vocab.id2token)

    extra = preprocess_for_char_embeddings("abcx";
                                           vocab      = orig_vocab,
                                           id_options = Dict(:add_new => true))

    @test length(orig_vocab.id2token) == orig_len + 1       # x appended
    new_id = orig_vocab.token2id["x"]
    @test orig_vocab.counts[new_id] == 1                    # counts updated
    @test extra.char_ids[end] == new_id


    
    #min_freq filter in vocab_options
    txt_lowfreq = "aaab"                                   # 'a' freq=3, 'b' freq=1
    vfilt = preprocess_for_char_embeddings(txt_lowfreq;
                                           vocab_options = Dict(:min_freq => 2))

    @test haskey(vfilt.vocabulary.token2id, "a")            # kept
    @test !haskey(vfilt.vocabulary.token2id, "b")           # filtered out
    @test vfilt.char_ids[end] == vfilt.vocabulary.unk_id    #'b' - <unk>


    #tmp-file cleanup guard
    tmp_path = ""
    mktemp() do path, io
        tmp_path = path
        write(io, "Test")
        close(io)
        preprocess_for_char_embeddings(path)    # just exercise the call
    end
    @test !isfile(tmp_path)                     # mktemp has removed the file
end



@testset "preprocess_for_char_embeddings - edge cases" begin

    #mixed accents + uppercase transform + accent-stripping
    txt_acc = "Áaá\n"
    out_acc = preprocess_for_char_embeddings(txt_acc;
                                            clean_options = Dict(
                                                :case_transform     => :upper,
                                                :do_remove_accents  => true,
                                                :unicode_normalize  => true
                                            ))

    expected_clean = "AAA"                     # three letters, newline removed
    expected_chars = ["A","A","A"]

    @test out_acc.cleaned_text == expected_clean
    @test out_acc.chars == expected_chars
    @test out_acc.char_ids == [out_acc.vocabulary.token2id["A"] for _ in 1:3]


    
    #user-supplied special tokens (deduplicated, kept up-front)
    
    res_spec = preprocess_for_char_embeddings("xy";
                    vocab_options = Dict(:special_tokens => ["<pad>", "<unk>"]))

    @test res_spec.vocabulary.id2token[1:2] == ["<pad>", "<unk>"]
    # no duplicates even if we supply <unk> again
    res_spec2 = preprocess_for_char_embeddings("xy";
                     vocab_options = Dict(:special_tokens => ["<unk>", "<unk>"]))
    @test res_spec2.vocabulary.id2token[1] == "<unk>" &&
          length(res_spec2.vocabulary.id2token[1:2]) == 2   # some non-<unk> token occupies slot 2



    
    #update_counts = false leaves counts unchanged
    
    base2   = preprocess_for_char_embeddings("zzz")
    vocab2  = base2.vocabulary
    counts0 = deepcopy(vocab2.counts)

    preprocess_for_char_embeddings("zzz";
                                   vocab       = vocab2,
                                   id_options  = Dict(:update_counts => false))

    @test vocab2.counts == counts0              # nothing incremented


    
    #unknown character mapped to <unk> when add_new = false
    res_unk = preprocess_for_char_embeddings("§"; vocab = vocab2,
                                             id_options = Dict(:add_new => false))
    @test res_unk.char_ids[1] == vocab2.unk_id


    #large min_freq: all rarities - <unk>
    big_txt = "abcabcabcXYZ"     # X,Y,Z each frequency 1
    res_freq = preprocess_for_char_embeddings(big_txt;
                    vocab_options = Dict(:min_freq => 3))

    @test !haskey(res_freq.vocabulary.token2id, "X")
    @test res_freq.char_ids[end] == res_freq.vocabulary.unk_id
end



@testset "preprocess_for_char_embeddings - large corpus smoke-test" begin
    #build a ~250 kB synthetic corpus and persist it
    
    sentence   = "The quick brown fox jumps over the lazy dog. "
    big_text   = repeat(sentence, 5000)              #225 kB
    
    path, io = mktemp()                              # path first, io second
    write(io, big_text)
    close(io)

    #run the preprocessing pipeline on that file
    
    res = preprocess_for_char_embeddings(
              path;
              clean_options = Dict(:case_transform => :lower),
              vocab_options = Dict(:min_freq => 100)
          )

    
    #integrity checks
    #cleaned text is non-empty and far shorter than raw only because
    # spaces were collapsed - not because the file vanished
    @test length(res.cleaned_text) > 100_000

    #consistent lengths: ids == chars
    @test length(res.char_ids) == length(res.chars)

    #vocabulary should contain more than just <unk>
    @test length(res.vocabulary.id2token) > 10

    #unseen glyph maps to <unk>
    unk_res = preprocess_for_char_embeddings("§"; vocab=res.vocabulary,
                                             id_options=Dict(:add_new=>false))
    @test unk_res.char_ids[1] == res.vocabulary.unk_id

    
    #slice into windows and sanity-check
    function windowify(ids::Vector{Int}, win::Int, stride::Int)
        [ids[i:i+win-1] for i in 1:stride:length(ids)-win+1]
    end
    windows = windowify(res.char_ids, 128, 64)
    @test !isempty(windows)
    @test all(length(w) == 128 for w in windows)
    
    #clean up temp-file
    rm(path; force=true)
    @test !isfile(path)
end



@testset "preprocess_for_char_embeddings - real text download" begin
    #download Alice's Adventures in Wonderland (150 kB)
    
    url  = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
    path = tempname() * ".txt"

    try
        Downloads.download(url, path)
    catch e
        @info "Network unavailable - skipping download test" exception = e
        return                          # skip the entire test-set
    end

    #preprocess the downloaded file
    out = preprocess_for_char_embeddings(
              path;
              clean_options = Dict(:case_transform => :lower),
              vocab_options = Dict(:min_freq => 10)      # keep common chars
          )

    
    #logic checks
    @test length(out.cleaned_text) > 100_000
    @test length(out.chars) == count(!isspace, out.cleaned_text) 
    @test all(c in keys(out.vocabulary.token2id) for c in ["a","e","t"]) 

    res_unk = preprocess_for_char_embeddings("§";
                  vocab = out.vocabulary,
                  id_options = Dict(:add_new => false))
    @test res_unk.char_ids[1] == out.vocabulary.unk_id   # 3d

    #clean up
    rm(path; force = true)
    @test !isfile(path)
end
