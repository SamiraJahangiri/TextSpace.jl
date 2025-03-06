

function remove_punctuation(
    text::String; 
    remove_symbols::Bool=false, 
    extra_symbols::Vector{Char}=[]
)
    if remove_symbols #remove punctuation + symbols, then any additional user-specified chars
        text = replace(text, r"[\p{P}\p{S}]" => "")
    else #remove punctuation only
        text = replace(text, r"\p{P}" => "")
    end

    for sym in extra_symbols #Char => "" approach or a small custom replace to remove each
        text = replace(text, sym => "")
    end

    return text
end


function remove_emojis(text::String)
    is_emoji(c::Char) = begin
        cp = UInt32(c)
        return (cp >= 0x1F300 && cp <= 0x1F5FF)  ||  # Misc Symbols and Pictographs
               (cp >= 0x1F600 && cp <= 0x1F64F)  ||  # Emoticons
               (cp >= 0x1F680 && cp <= 0x1F6FF)  ||  # Transport/Map
               (cp >= 0x1F700 && cp <= 0x1F77F)  ||  # Alchemical Symbols, etc. (some are emojis)
               (cp >= 0x1F900 && cp <= 0x1F9FF)  ||  # Supplemental Symbols/Pictographs
               (cp >= 0x1FA70 && cp <= 0x1FAFF)  ||  # Symbols/Pictographs Extended-A
               (cp >= 0x2600 && cp <= 0x26FF)    ||  # Misc Symbols
               (cp >= 0x2700 && cp <= 0x27BF)    ||  # Dingbats
               (cp >= 0x1F1E6 && cp <= 0x1F1FF)       # Regional Indicator Symbols (flags)
    end
    return join(filter(c -> !is_emoji(c), text))
end


function remove_accents(text::String)
    #decompose (NFD)
    text_nfd = Unicode.normalize(text, :NFD)
    
    #remove nonspacing marks (\p{Mn}), strips diacritical marks while leaving base characters intact
    text_no_diacritics = replace(text_nfd, r"\p{Mn}" => "")
    
    #recompose (NFC) to avoid leftover decomposed characters
    text_clean = Unicode.normalize(text_no_diacritics, :NFC)

    return text_clean
end


function clean_text(
    text::String;
    unicode_normalize::Bool=true,
    remove_accents::Bool=false,
    remove_punctuation::Bool=false,
    remove_symbols::Bool=false,
    remove_emojis::Bool=false,
    extra_symbols::Vector{Char}=[]
)
    #Unicode normalization (NFC) to standardize form
    if unicode_normalize
        text = normalize_unicode(text)
    end

    #remove accents/diacritics
    if remove_accents
        text = remove_accents(text)
    end

    #convert to lowercase (often done early, but after unicode normalize is safe)
    text = lowercase(text)

    #remove punctuation (and possibly symbols)
    if remove_punctuation || remove_symbols || !isempty(extra_symbols)
        text = remove_punctuation(
            text;
            remove_symbols=remove_symbols,
            extra_symbols=extra_symbols
        )
    end

    #remove emojis
    if remove_emojis
        text = remove_emojis(text)
    end

    #normalize whitespace (collapse multiple spaces, strip ends)
    text = normalize_whitespace(text)

    return text
end
