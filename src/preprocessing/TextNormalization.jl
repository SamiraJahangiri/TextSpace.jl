# (Optional) Additional text normalization (e.g., accent stripping, stemming, lemmatization)

"""
    normalize_unicode(text; form = :NFC) -> String

Return *text* in the requested Unicode normal-form (`:NFC`, `:NFD`, `:NFKC`, …).
"""
normalize_unicode(text::String; form::Symbol = :NFC) =
    Unicode.normalize(text, form)


"""
    normalize_whitespace(text;
                         strip_ends       = true,
                         preserve_newlines = false,
                         remove_zero_width = false) -> String

* Collapse runs of whitespace to a single space  
  – if `preserve_newlines = true`, new-line characters are kept while
  spaces/tabs/carriage-returns are collapsed.

* Optionally strip leading / trailing blanks (`strip_ends = true`).

* Optionally remove common zero-width code-points.

The helper is UTF-8 safe and leaves non-ASCII scripts untouched.
"""
function normalize_whitespace(text::AbstractString;
                              strip_ends::Bool        = true,
                              preserve_newlines::Bool = false,
                              remove_zero_width::Bool = false)

    t = text                     # local working copy

   
    if remove_zero_width
        t = replace(t, r"[\u200B\u200C\u200D\uFEFF]+" => "")
    end

    if preserve_newlines
        # keep '\n', but squeeze other blanks (space, tab, CR, FF) to one
        t = replace(t, r"[ \t\f\r]+" => " ")
        # trim any spaces that now sit *before* a newline (handles CR-LF too)
        t = replace(t, r" +\n" => "\n")
    else
        # replace *any* run of Unicode whitespace with a single space
        t = replace(t, r"\s+" => " ")
    end

    strip_ends && (t = strip(t))

    return t
end