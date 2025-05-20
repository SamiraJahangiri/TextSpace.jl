function windowify(ids::AbstractVector{Int}; win::Int, stride::Int)
    last = length(ids) - win + 1            # same formula as character code
    last <= 0 && return Vector{Vector{Int}}()

    # keep only full windows of length `win`
    return [ @view ids[i : i+win-1] for i in 1:stride:last ]
end