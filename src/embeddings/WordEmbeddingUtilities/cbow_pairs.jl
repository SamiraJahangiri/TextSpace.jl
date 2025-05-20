

function cbow_pairs(win::AbstractVector{Int}, radius::Int)
    pairs = Tuple{Vector{Int},Int}[]
    for i in radius+1 : length(win)-radius
        ctx = collect(view(win, i-radius:i+radius))
        deleteat!(ctx, radius+1)           # drop the centre
        push!(pairs, (ctx, win[i]))
    end
    return pairs
end