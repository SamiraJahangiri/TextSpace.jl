"""
    windowify(ids, win, stride)

Return overlapping windows of length `win` taken every `stride` steps.
"""
function windowify(ids::Vector{Int}, win::Int, stride::Int)
    last = length(ids) - win + 1
    last <= 0 && return Vector{Vector{Int}}()
    [@view(ids[i:i+win-1]) for i in 1:stride:last]   # uses views
end