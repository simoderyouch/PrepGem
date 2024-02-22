from lexiclean import LexiClean
cleaner = LexiClean()

# Example usage
text = "This is an exampl text with  tags and URLs  fwefwe egergfwaawaffa  tags"

print(cleaner.clean_and_spell_check_text(text))
