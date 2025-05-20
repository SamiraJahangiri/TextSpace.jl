
"""
    skipgram_pairs(win_ids, radius)

Generate (center, context) index pairs within ±`radius` of each position.
"""
function skipgram_pairs(win::AbstractVector{Int}, radius::Int)
    pairs = Tuple{Int,Int}[]
    for i in eachindex(win)
        for j in max(1,i-radius):min(length(win), i+radius)
            i == j && continue
            push!(pairs, (win[i], win[j]))
        end
    end
    return pairs
end