# include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Preprocessing.jl"))



# @testset "preprocess_for_char_embeddings" begin
    
#     #raw string - new vocabulary
    
#     txt  = "Hello 😊"
#     out  = preprocess_for_char_embeddings(txt)          # build vocab

#     #cleaning left the visible text unchanged
#     @test out.cleaned_text == txt

#     #tokenisation: default keeps case, drops spaces
#     @test out.chars == ["H","e","l","l","o"," ","😊"]

#     # <unk> exists and has a valid positive id
#     @test haskey(out.vocabulary.token2id, "<unk>")
#     @test out.vocabulary.unk_id >= 1

#     #char_ids are a 1-to-1 mapping of the returned characters
#     @test out.char_ids ==
#           [out.vocabulary.token2id[c] for c in out.chars]

    
#     #re-using an existing vocabulary
#     txt2 = "Hola"
#     out2 = preprocess_for_char_embeddings(txt2; vocab = out.vocabulary)

#     #should not create a brand-new Vocabulary object
#     @test out2.vocabulary === out.vocabulary

#     #all ids are within the known range
#     @test all(1 ≤ id ≤ length(out.vocabulary.id2token) for id in out2.char_ids)

    
#     #file-path input & unknown characters fall back to <unk>
#     mktemp() do path, io          # ← path first, stream second
#         write(io, "¿Qué?")
#         close(io)
    
#         out3 = preprocess_for_char_embeddings(path; vocab = out.vocabulary)
    
#         # first character should be mapped to the unk id
#         @test out3.char_ids[1] == out.vocabulary.unk_id
#     end
# end


# @testset "preprocess_for_char_embeddings - additional coverage" begin
    
#     #clean_options + char_options flags
#     txt = "Go :)"
#     res = preprocess_for_char_embeddings(txt;
#                                         clean_options = Dict(
#                                             :case_transform        => :lower,
#                                             :do_remove_punctuation => true
#                                         ),
#                                         char_options  = Dict(:keep_space => true))

#     @test res.cleaned_text == "go"               # no trailing blank after normalisation
#     @test res.chars == ["g","o"]
#     @test res.char_ids[1:2] ==
#         [res.vocabulary.token2id[c] for c in res.chars]     # length 2



#     # add_new = true really grows the vocabulary and updates counts
#     base = preprocess_for_char_embeddings("abc")    # fresh vocab
#     orig_vocab = base.vocabulary
#     orig_len   = length(orig_vocab.id2token)

#     extra = preprocess_for_char_embeddings("abcx";
#                                            vocab      = orig_vocab,
#                                            id_options = Dict(:add_new => true))

#     @test length(orig_vocab.id2token) == orig_len + 1       # x appended
#     new_id = orig_vocab.token2id["x"]
#     @test orig_vocab.counts[new_id] == 1                    # counts updated
#     @test extra.char_ids[end] == new_id


    
#     #min_freq filter in vocab_options
#     txt_lowfreq = "aaab"                                   # 'a' freq=3, 'b' freq=1
#     vfilt = preprocess_for_char_embeddings(txt_lowfreq;
#                                            vocab_options = Dict(:min_freq => 2))

#     @test haskey(vfilt.vocabulary.token2id, "a")            # kept
#     @test !haskey(vfilt.vocabulary.token2id, "b")           # filtered out
#     @test vfilt.char_ids[end] == vfilt.vocabulary.unk_id    #'b' - <unk>


#     #tmp-file cleanup guard
#     tmp_path = ""
#     mktemp() do path, io
#         tmp_path = path
#         write(io, "Test")
#         close(io)
#         preprocess_for_char_embeddings(path)    # just exercise the call
#     end
#     @test !isfile(tmp_path)                     # mktemp has removed the file
# end


# @testset "preprocess_for_char_embeddings - edge cases" begin

#     #mixed accents + uppercase transform + accent-stripping
#     txt_acc = "Áaá\n"
#     out_acc = preprocess_for_char_embeddings(txt_acc;
#                                             clean_options = Dict(
#                                                 :case_transform     => :upper,
#                                                 :do_remove_accents  => true,
#                                                 :unicode_normalize  => true
#                                             ))

#     expected_clean = "AAA"                     # three letters, newline removed
#     expected_chars = ["A","A","A"]

#     @test out_acc.cleaned_text == expected_clean
#     @test out_acc.chars == expected_chars
#     @test out_acc.char_ids == [out_acc.vocabulary.token2id["A"] for _ in 1:3]


    
#     #user-supplied special tokens (deduplicated, kept up-front)
    
#     res_spec = preprocess_for_char_embeddings("xy";
#                     vocab_options = Dict(:special_tokens => ["<pad>", "<unk>"]))

#     @test res_spec.vocabulary.id2token[1:2] == ["<pad>", "<unk>"]
#     # no duplicates even if we supply <unk> again
#     res_spec2 = preprocess_for_char_embeddings("xy";
#                      vocab_options = Dict(:special_tokens => ["<unk>", "<unk>"]))
#     @test res_spec2.vocabulary.id2token[1] == "<unk>" &&
#           length(res_spec2.vocabulary.id2token[1:2]) == 2   # some non-<unk> token occupies slot 2



    
#     #update_counts = false leaves counts unchanged
    
#     base2   = preprocess_for_char_embeddings("zzz")
#     vocab2  = base2.vocabulary
#     counts0 = deepcopy(vocab2.counts)

#     preprocess_for_char_embeddings("zzz";
#                                    vocab       = vocab2,
#                                    id_options  = Dict(:update_counts => false))

#     @test vocab2.counts == counts0              # nothing incremented


    
#     #unknown character mapped to <unk> when add_new = false
#     res_unk = preprocess_for_char_embeddings("§"; vocab = vocab2,
#                                              id_options = Dict(:add_new => false))
#     @test res_unk.char_ids[1] == vocab2.unk_id


#     #large min_freq: all rarities - <unk>
#     big_txt = "abcabcabcXYZ"     # X,Y,Z each frequency 1
#     res_freq = preprocess_for_char_embeddings(big_txt;
#                     vocab_options = Dict(:min_freq => 3))

#     @test !haskey(res_freq.vocabulary.token2id, "X")
#     @test res_freq.char_ids[end] == res_freq.vocabulary.unk_id
# end


# @testset "preprocess_for_char_embeddings - large corpus smoke-test" begin
#     #build a approx 250 kB synthetic corpus and persist it
    
#     sentence   = "The quick brown fox jumps over the lazy dog. "
#     big_text   = repeat(sentence, 5000)              #225 kB
    
#     path, io = mktemp()                              # path first, io second
#     write(io, big_text)
#     close(io)

#     #run the preprocessing pipeline on that file
    
#     res = preprocess_for_char_embeddings(
#               path;
#               clean_options = Dict(:case_transform => :lower),
#               vocab_options = Dict(:min_freq => 100)
#           )

    
#     #integrity checks
#     #cleaned text is non-empty and far shorter than raw only because
#     # spaces were collapsed - not because the file vanished
#     @test length(res.cleaned_text) > 100_000

#     #consistent lengths: ids == chars
#     @test length(res.char_ids) == length(res.chars)

#     #vocabulary should contain more than just <unk>
#     @test length(res.vocabulary.id2token) > 10

#     #unseen glyph maps to <unk>
#     unk_res = preprocess_for_char_embeddings("§"; vocab=res.vocabulary,
#                                              id_options=Dict(:add_new=>false))
#     @test unk_res.char_ids[1] == res.vocabulary.unk_id

    
#     #slice into windows and sanity-check
#     function windowify(ids::Vector{Int}, win::Int, stride::Int)
#         [ids[i:i+win-1] for i in 1:stride:length(ids)-win+1]
#     end
#     windows = windowify(res.char_ids, 128, 64)
#     @test !isempty(windows)
#     @test all(length(w) == 128 for w in windows)
    
#     #clean up temp-file
#     rm(path; force=true)
#     @test !isfile(path)
# end


# @testset "preprocess_for_char_embeddings - real text download" begin
#     #download Alice's Adventures in Wonderland (150 kB)
    
#     url  = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
#     path = tempname() * ".txt"

#     try
#         Downloads.download(url, path)
#     catch e
#         @info "Network unavailable - skipping download test" exception = e
#         return                          # skip the entire test-set
#     end

#     #preprocess the downloaded file
#     out = preprocess_for_char_embeddings(
#               path;
#               clean_options = Dict(:case_transform => :lower),
#               vocab_options = Dict(:min_freq => 10)      # keep common chars
#           )

    
#     #logic checks
#     @test length(out.cleaned_text) > 100_000
#     @test length(out.chars) == length(Unicode.graphemes(out.cleaned_text))
#     @test all(c in keys(out.vocabulary.token2id) for c in ["a","e","t"]) 

#     res_unk = preprocess_for_char_embeddings("§";
#                   vocab = out.vocabulary,
#                   id_options = Dict(:add_new => false))
#     @test res_unk.char_ids[1] == out.vocabulary.unk_id   # 3d

#     #clean up
#     rm(path; force = true)
#     @test !isfile(path)
# end


# @testset "preprocess_for_char_embeddings - option matrix" begin
#     #cleaning / whitespace / punctuation / accent flags
#     raw = "Café   \t\n🚀!!  "    # accents, repeated blanks, emoji, punct

#     clean_opts = Dict(
#         :case_transform        => :lower,
#         :do_remove_punctuation => true,
#         :do_remove_accents     => true,
#         :collapse_whitespace   => true,     # turn runs of blanks -> one space
#     )
#     char_opts  = Dict(:keep_space => false) # drop the single space we kept
#     outA = preprocess_for_char_embeddings(raw;
#                 from_file      = false,
#                 clean_options  = clean_opts,
#                 char_options   = char_opts)

#     @test outA.cleaned_text == "cafe 🚀"          # accent stripped, blanks to 1, punct gone
#     @test outA.chars == ["c","a","f","e","🚀"]    # no space token
#     @test length(outA.char_ids) == 5

#     #space-keeping + Unicode-normalisation left intact
#     raw2 = "Fiancée " * "\u202F" * "Ωmega"        # NARROW NBSP between words
#     outB = preprocess_for_char_embeddings(raw2;
#                 clean_options = Dict(:unicode_normalize => true),  # NFC default
#                 char_options  = Dict(:keep_space => true))

#     @test " " in outB.chars                      # space token kept
#     @test occursin("fiancée", lowercase(outB.cleaned_text))  # NFC preserved é

#     #external vocabulary + add_new / update_counts flags
#     # make a tiny vocab with <unk>, a, b
#     tok2id = Dict("<unk>"=>1, "a"=>2, "b"=>3)
#     id2tok = ["<unk>","a","b"]
#     extvoc = TextSpace.Preprocessing.Vocabulary(tok2id, id2tok, Dict{Int,Int}(), 1)

#     txtC = "abx"    # 'x' is OOV
#     outC1 = preprocess_for_char_embeddings(txtC;
#                  vocab       = extvoc,
#                  char_options=Dict(:keep_space=>false),
#                  id_options  = Dict(:add_new=>false, :update_counts=>false))

#     @test outC1.char_ids == [2,3,1]          # x -> unk_id
#     @test !haskey(extvoc.token2id, "x")      # vocab unchanged

#     # same text, but now allow growth and counting
#     outC2 = preprocess_for_char_embeddings(txtC;
#                  vocab       = extvoc,
#                  id_options  = Dict(:add_new=>true,  :update_counts=>true))

#     @test extvoc.token2id["x"] == length(extvoc.id2token)  # new token inserted
#     @test outC2.char_ids[end] == extvoc.token2id["x"]
#     @test extvoc.counts[ extvoc.token2id["x"] ] == 1       # counts updated

#     #file-path input (temp file) + ensure_unk! auto-repairs
#     mktemp() do path, io
#         write(io, "§")             # char not in extvoc
#         close(io)

#         broken = TextSpace.Preprocessing.Vocabulary(Dict("a"=>1), ["a"], Dict{Int,Int}(), 0)
#         outD   = preprocess_for_char_embeddings(path;
#                     from_file = true,
#                     vocab     = broken,            # will create *new* vocab
#                     id_options= Dict(:add_new=>false))

#         #the pipeline returns a *new* repaired vocabulary
#         @test outD.vocabulary !== broken
#         @test outD.vocabulary.unk_id >= 1
#         @test outD.char_ids[1] == outD.vocabulary.unk_id


#         rm(path; force=true)
#     end
# end


# @testset "preprocess_for_char_embeddings - full option sweep" begin
#     #cleaning + whitespace + accent/punct/emoji removal
#     raw = "Café   \t\n🚀!!  —  Ωmega🙂"

#     clean_opts = Dict(
#         :case_transform        => :lower,
#         :do_remove_punctuation => true,
#         :do_remove_symbols     => true,
#         :do_remove_emojis      => true,
#         :do_remove_accents     => true,
#         :collapse_whitespace   => true,   # collapse runs - single space
#     )
#     char_opts = Dict(:keep_space => false)
#     outA = preprocess_for_char_embeddings(raw;
#                 clean_options = clean_opts,
#                 char_options  = char_opts,
#                 from_file     = false)

#     @test occursin("cafe ωmega", outA.cleaned_text)          # accent stripped, lower-cased
#     @test !occursin('🚀', outA.cleaned_text) && !occursin('🙂', outA.cleaned_text)
#     @test !occursin('—', outA.cleaned_text)                  # em-dash removed by punctuation flag
#     @test " " ∉ outA.chars                                   # because keep_space=false
#     @test length(outA.char_ids) == length(outA.chars)

#     #keep_space = true + NFC normalisation only
#     raw2 = "Fiancée " * "\u202F" * "Ωmega"   # NARROW NBSP between words
#     outB = preprocess_for_char_embeddings(raw2;
#                 clean_options = Dict(:unicode_normalize => true, :case_transform=>:lower),
#                 char_options  = Dict(:keep_space => true),
#                 from_file     = false)

#     @test " " in outB.chars
#     @test occursin("fiancée ωmega", outB.cleaned_text)

#     #external vocabulary + add_new / update_counts
#     tok2id = Dict("<unk>"=>1, "a"=>2, "b"=>3)
#     id2tok = ["<unk>","a","b"]
#     extvoc = TextSpace.Preprocessing.Vocabulary(tok2id, id2tok, Dict{Int,Int}(), 1)

#     # add_new=false  keeps OOV as unk
#     r1 = preprocess_for_char_embeddings("abx";
#             vocab      = extvoc,
#             id_options = Dict(:add_new=>false, :update_counts=>false),
#             char_options = Dict(:keep_space=>false))
#     @test r1.char_ids == [2,3,1]
#     @test !haskey(extvoc.token2id, "x")

#     # add_new=true  extends vocab and updates counts
#     r2 = preprocess_for_char_embeddings("abx";
#             vocab      = extvoc,
#             id_options = Dict(:add_new=>true, :update_counts=>true))
#     new_id = extvoc.token2id["x"]
#     @test r2.char_ids[end] == new_id
#     @test extvoc.counts[new_id] == 1

#     #file input + ensure_unk! auto-repair
#     mktemp() do path, io
#         write(io, "§"); close(io)

#         broken = TextSpace.Preprocessing.Vocabulary(Dict("a"=>1), ["a"],
#                                                     Dict{Int,Int}(), 0)   # unk_id = 0

#         outD = preprocess_for_char_embeddings(path;
#                     from_file = true,
#                     vocab     = broken,
#                     id_options = Dict(:add_new=>false))

#         @test outD.vocabulary !== broken   # got a repaired copy
#         @test outD.vocabulary.unk_id ≥ 1
#         @test outD.char_ids[1] == outD.vocabulary.unk_id

#         rm(path; force=true)
#     end

#     #min_freq filtering
#     r5 = preprocess_for_char_embeddings("xxxyyZ";
#             vocab_options = Dict(:min_freq=>2))
#     @test !haskey(r5.vocabulary.token2id, "Z")
#     @test r5.char_ids[end] == r5.vocabulary.unk_id
# end


# @testset "preprocess_for_char_embeddings - curated UTF-8 hammer" begin
#     #paragraph: 8 sentences with new-lines, tabs, ZWSP, emoji,
#     #  combining marks, bidi controls, ligatures, NBSP, narrow NBSP,
#     #  and a zero-width joiner sequence
#     zwsp   = "\u200B"                       # ZERO-WIDTH SPACE
#     nbsp   = "\u00A0"                       # NBSP
#     nnbsp  = "\u202F"                       # NARROW NBSP
#     rle    = "\u202B"                       # RTL EMBEDDING
#     pdf    = "\u202C"                       # POP DIR. FORMAT
#     ligfi  = "ﬁ"
#     combé  = "e\u0301"                      # e + COMBINING ACUTE
#     famemo = "👨‍👩‍👧‍👦"                    # family emoji with ZWJ
#     astro  = "👩🏽‍🚀"
#     para = """
#     Once upon a  time,\tthere were two cafés.$(nbsp)$(nbsp)
#     They said: “$(ligfi)\u200Breﬂies?!  No way!”\n
#     Meanwhile, 数学 is fun; $nnbsp but $(rle)مرحبا بالعالم$(pdf) was written backwards.
#     Tabs,  spaces,\n\nnew-lines, and $zwsp zero-widths $zwsp hide! $astro went to the 🌖.
#     $famemo danced in the night… $(combé)!
#     """

#     #conservative cleaning (NFC only) + keep spaces
#     outA = preprocess_for_char_embeddings(para;
#             clean_options = Dict(:unicode_normalize=>true),
#             char_options  = Dict(:keep_space=>true),
#             from_file     = false)

#     @test occursin("café", outA.cleaned_text)      # accent preserved
#     @test '🌖' in outA.cleaned_text
#     @test '\n' ∉ outA.cleaned_text                 # normalize_whitespace default
#     @test " " in outA.chars                        # spaces kept
#     @test length(outA.chars) == length(Unicode.graphemes(outA.cleaned_text))

#     #aggressive emoji + punctuation + accent removal, collapse whitespace, drop spaces
#     outB = preprocess_for_char_embeddings(para;
#             clean_options = Dict(
#                 :do_remove_emojis      => true,
#                 :do_remove_punctuation => true,
#                 :do_remove_accents     => true,
#                 :collapse_whitespace   => true,
#                 :case_transform        => :lower),
#             char_options  = Dict(:keep_space=>false),
#             from_file     = false)

#     @test !occursin('🌖', outB.cleaned_text) && !occursin('👨', outB.cleaned_text)
#     @test !occursin("¡", outB.cleaned_text)        # punctuation gone
#     @test !occursin("é", outB.cleaned_text)        # accent stripped
#     @test !occursin(r"\s\s", outB.cleaned_text)    # no double blanks
#     @test " " ∉ outB.chars                         # spaces dropped
#     @test outB.cleaned_text == lowercase(outB.cleaned_text)

#     #symbols removed but punctuation kept; case upper
#     outC = preprocess_for_char_embeddings(para;
#             clean_options = Dict(
#                 :do_remove_symbols     => true,   # removes currency, math, emoji
#                 :do_remove_emojis      => false,  # but we already stripped symbols
#                 :case_transform        => :upper),
#             char_options = Dict(:keep_space=>true),
#             from_file    = false)

#     @test occursin("CAFÉ", outC.cleaned_text)
#     @test '🌖' ∉ outC.cleaned_text                 # symbol removed
#     @test 'É' ∈ outC.cleaned_text                 # accent still there
#     @test any(c -> isuppercase(c[1]), outC.chars) # uppercase present

#     #high min_freq filters rare glyphs; ensure OOV-><unk>
#     rare_opts = Dict(:min_freq => 10)
#     rD  = preprocess_for_char_embeddings(para;
#             vocab_options = rare_opts,
#             char_options  = Dict(:keep_space=>false),
#             from_file     = false)

#     #rare glyph '🌖' should NOT be in the pruned vocabulary
#     @test !haskey(rD.vocabulary.token2id, "🌖")

#     #every occurrence in the corpus must therefore map to <unk>
#     @test all(id == rD.vocabulary.unk_id
#             for (tok,id) in zip(rD.chars, rD.char_ids) if tok == "🌖")

# end
