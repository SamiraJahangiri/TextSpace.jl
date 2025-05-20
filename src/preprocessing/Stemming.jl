function stem_text(text::String; language::String="english")
    words = split(text) 
    stemmed_words = [porter_stem(w) for w in words] 
    return join(stemmed_words, ' ') 
end

function is_consonant(w::String, i::Int) 
    c = w[i] 
    return !( c in ('a','e','i','o','u') || (c=='y' && i>1 && !(w[i-1] in ('a','e','i','o','u'))) ) 
end

function measure(w::String) 
    m = 0 
    in_vowel_seq = false 
    for i in eachindex(w) 
        if is_consonant(w, i) 
            if in_vowel_seq m += 1 
            end 
            in_vowel_seq = false 
        else 
            in_vowel_seq = true 
        end 
    end 
    return m 
end



function replace_suffix(w::String, suffix::String, repl::String) 
    return w[1:end-length(suffix)] * repl 
end

function step1(w::String)
    # Step 1a: handle plurals
    if endswith(w, "sses")
        w = replace_suffix(w, "sses", "ss")
    elseif endswith(w, "ies")
        w = replace_suffix(w, "ies", "i")
    elseif endswith(w, "ss")
        # do nothing
    elseif endswith(w, "s")
        w = w[1:end-1]
    end

    # Step 1b: -ed or -ing
    if endswith(w, "ed") && has_vowel(w[1:end-2])
        w = w[1:end-2]
        w = step1b_helper(w)
    elseif endswith(w, "ing") && has_vowel(w[1:end-3])
        w = w[1:end-3]
        w = step1b_helper(w)
    end

    # Step 1c: y with ible
    if endswith(w, "y") && has_vowel(w[1:end-1])
        w = replace_suffix(w, "y", "i")
    end

    return w
end


function step1b_helper(w::String) 
    if endswith(w, "at") 
        w = w * "e" 
    elseif endswith(w, "bl") 
        w = w * "e" 
    elseif endswith(w, "iz") 
        w = w * "e" 
    elseif double_consonant(w) 
        w = w[1:end-1] 
    elseif measure(w) == 1 && cvc(w) 
        w = w * "e" 
    end 
    return w 
end

function has_vowel(w::String) 
    for c in w 
        if c in ('a','e','i','o','u') 
            return true 
        end 
    end 
    return false 
end


function double_consonant(w::String)
    n = length(w)
    if n < 2
        return false
    end
    return is_consonant(w, n) && w[n] == w[n-1]
end


function cvc(w::String) 
    if length(w) < 3 
        return false 
    end 
    last3 = w[end-2:end] 
    return is_consonant(last3,1) && !(is_consonant(last3,2)) && is_consonant(last3,3) && !(last3[end] in ('w','x','y')) 
end

function step2(w::String) #common suffix replacements for measure(w) > 0 
    step2_rules = [ ("ational","ate"), ("tional","tion"), ("enci","ence"), ("anci","ance"), ("izer","ize"), ("bli","ble"), ("alli","al"), ("entli","ent"), ("eli","e"), ("ousli","ous"), ("ization","ize"), ("ation","ate"), ("ator","ate"), ("alism","al"), ("iveness","ive"), ("fulness","ful"), ("ousness","ous"), ("aliti","al"), ("iviti","ive"), ("biliti","ble") ] 
    
    for (old, new) in step2_rules 
        if endswith(w, old) && measure(w[1:end-length(old)]) > 0 
            return replace_suffix(w, old, new) 
        end 
    end 

    return w 
end

function step3(w::String) 
    step3_rules = [ ("icate","ic"), ("ative",""), ("alize","al"), ("iciti","ic"), ("ical","ic"), ("ful",""), ("ness","") ] 
    for (old, new) in step3_rules 
        if endswith(w, old) && measure(w[1:end-length(old)]) > 0 
            return replace_suffix(w, old, new) 
        end 
    end 
    return w 
end

function step4(w::String) 
    step4_rules = ["al","ance","ence","er","ic","able","ible","ant","ement", "ment","ent","ou","ism","ate","iti","ous","ive","ize"] 
    for old in step4_rules 
        if endswith(w, old) && measure(w[1:end-length(old)]) > 1 
            return w[1:end-length(old)] 
        end 
    end #special case: "ion" 
    if endswith(w, "ion") && measure(w[1:end-3]) > 1 && (last(w[1:end-3]) in ['s','t']) 
        return w[1:end-3] 
    end 
    return w 
end

function step5(w::String) 
    #step 5a 
    if endswith(w, "e") 
        if measure(w[1:end-1]) > 1 
            w = w[1:end-1] 
        elseif measure(w[1:end-1]) == 1 && !cvc(w[1:end-1]) 
            w = w[1:end-1] 
        end 
    end #step 5b 
    if measure(w) > 1 && double_consonant(w) && last(w) == 'l' 
        w = w[1:end-1] 
    end 
    return w 
end


function porter_stem(word::AbstractString)
    w = lowercase(word)

    w = step1(w)
    w = step2(w)
    w = step3(w)
    w = step4(w)
    w = step5(w)

    return w
end
