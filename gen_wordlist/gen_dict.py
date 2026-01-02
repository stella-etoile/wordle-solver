with open("words_alpha.txt", "r", encoding="utf-8") as f:
    words = [w.strip().lower() for w in f if len(w.strip()) == 5 and w.strip().isalpha()]

words = sorted(set(words))

with open("dictionary.txt", "w", encoding="utf-8") as f:
    for w in words:
        f.write(w + "\n")
