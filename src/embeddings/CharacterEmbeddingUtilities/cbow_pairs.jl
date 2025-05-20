function cbow_pairs(win::AbstractVector{Int}, radius::Int)
    pairs = Tuple{Vector{Int},Int}[]
    for i in radius+1:length(win)-radius          # centre has full context
        ctx = collect(view(win, i-radius:i+radius))
        deleteat!(ctx, radius+1)                  # drop the centre token
        push!(pairs, (ctx, win[i]))               # (context-vec, centre-id)
    end
    return pairs
end