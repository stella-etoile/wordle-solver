import math
from collections import Counter
import os
import multiprocessing

_ENTROPY_CANDIDATES = None

def load_words(path):
    words = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.append(w)
    except FileNotFoundError:
        print(f"Could not find {path}")
        raise
    if not words:
        raise ValueError(f"No words loaded from {path}")
    length = len(words[0])
    for w in words:
        if len(w) != length:
            raise ValueError("All words in dictionary must have same length")
    return words

def load_starters(path, allowed_words):
    starters = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    if w not in allowed_words:
                        raise ValueError(f"Starter word {w} not in dictionary")
                    starters.append(w)
    except FileNotFoundError:
        return []
    return starters

def pattern_for(secret, guess):
    n = len(secret)
    result = [0] * n
    secret_chars = list(secret)
    guess_chars = list(guess)
    for i in range(n):
        if guess_chars[i] == secret_chars[i]:
            result[i] = 2
            secret_chars[i] = None
            guess_chars[i] = None
    for i in range(n):
        if guess_chars[i] is not None:
            ch = guess_chars[i]
            if ch in secret_chars:
                result[i] = 1
                idx = secret_chars.index(ch)
                secret_chars[idx] = None
    return "".join(str(x) for x in result)

def refine_candidates(candidates, guess, pattern):
    return [w for w in candidates if pattern_for(w, guess) == pattern]

def entropy_for_guess(guess, possible_answers):
    total = len(possible_answers)
    counts = Counter()
    for ans in possible_answers:
        p = pattern_for(ans, guess)
        counts[p] += 1
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H

def _entropy_worker(guess):
    return guess, entropy_for_guess(guess, _ENTROPY_CANDIDATES)

def _entropy_init(candidates):
    global _ENTROPY_CANDIDATES
    _ENTROPY_CANDIDATES = candidates

def _print_progress_inline(label, i, total, last_state):
    if not label or total <= 0:
        return last_state
    pct = int(i * 100 / total)
    msg = f"{label} {pct:3d}% ({i}/{total})"
    print("\r" + msg, end="", flush=True)
    return msg

def _clear_progress_inline(last_msg):
    if not last_msg:
        return
    print("\r" + " " * len(last_msg), end="\r", flush=True)

def top_entropy_guesses(
    allowed_words,
    candidates,
    k=10,
    restrict_to_candidates=False,
    precomputed=None,
    n_jobs=1,
    progress_label=None,
):
    if restrict_to_candidates:
        pool_words = candidates
    else:
        pool_words = allowed_words
    if not pool_words:
        return []

    if precomputed is not None and len(candidates) == len(allowed_words):
        scores = []
        total = len(pool_words)
        last_msg = None
        for i, w in enumerate(pool_words, 1):
            H = precomputed.get(w, entropy_for_guess(w, candidates))
            scores.append((H, w))
            last_msg = _print_progress_inline(progress_label, i, total, last_msg)
        if last_msg:
            _clear_progress_inline(last_msg)
        scores.sort(reverse=True)
        return scores[:k] if k < len(scores) else scores

    use_parallel = n_jobs > 1 and len(pool_words) > 1
    scores = []
    total = len(pool_words)
    last_msg = None

    if use_parallel:
        with multiprocessing.Pool(
            processes=n_jobs,
            initializer=_entropy_init,
            initargs=(candidates,),
        ) as pool:
            for i, (w, H) in enumerate(pool.imap_unordered(_entropy_worker, pool_words), 1):
                scores.append((H, w))
                last_msg = _print_progress_inline(progress_label, i, total, last_msg)
    else:
        for i, w in enumerate(pool_words, 1):
            H = entropy_for_guess(w, candidates)
            scores.append((H, w))
            last_msg = _print_progress_inline(progress_label, i, total, last_msg)

    if last_msg:
        _clear_progress_inline(last_msg)
    scores.sort(reverse=True)
    return scores[:k] if k < len(scores) else scores

def ensure_first_entropy_cache(path, allowed_words, n_jobs):
    ent = {}
    word_set = set(allowed_words)

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    w, h = parts
                    if w in word_set:
                        ent[w] = float(h)
            if len(ent) == len(allowed_words):
                return ent
            else:
                print("Existing first-entropy cache incomplete; recomputing")
        except Exception:
            print("Error reading first-entropy cache; recomputing")

    print("Precomputing first-guess entropies…")
    total = len(allowed_words)
    n_jobs = n_jobs or (os.cpu_count() or 1)
    last_msg = None
    ent = {}
    with multiprocessing.Pool(
        n_jobs,
        initializer=_entropy_init,
        initargs=(allowed_words,),
    ) as pool:
        for i, (w, H) in enumerate(pool.imap_unordered(_entropy_worker, allowed_words), 1):
            ent[w] = H
            last_msg = _print_progress_inline("[first entropy]", i, total, last_msg)
    if last_msg:
        _clear_progress_inline(last_msg)
    print()

    with open(path, "w", encoding="utf-8") as f:
        for w in allowed_words:
            f.write(f"{w} {ent[w]:.8f}\n")

    print("First-turn entropies saved.")
    return ent

def print_candidates(cands, show=50):
    print(f"Possible answers left: {len(cands)}")
    if len(cands) <= show:
        print(" ".join(cands))
    else:
        print(" ".join(cands[:show]))

def mode_manual_assist(allowed_words, max_guesses, entropy_cache_path, n_jobs):
    word_len = len(allowed_words[0])
    cands = allowed_words[:]
    guess_count = 0
    first_ent = None
    while True:
        print()
        print_candidates(cands)
        print("\nTop high-entropy words:")
        if guess_count == 0 and len(cands) == len(allowed_words):
            if first_ent is None:
                first_ent = ensure_first_entropy_cache(entropy_cache_path, allowed_words, n_jobs)
            ranking = sorted(((H, w) for w, H in first_ent.items()), reverse=True)
            top10 = ranking[:10]
        else:
            top10 = top_entropy_guesses(
                allowed_words,
                cands,
                k=10,
                restrict_to_candidates=False,
                precomputed=None,
                n_jobs=n_jobs,
                progress_label="[manual entropy]",
            )
        for H, w in top10:
            mark = "*" if w in cands else ""
            print(f"{w}{mark}: {H:.4f}")

        if guess_count >= max_guesses:
            print("\nReached max guesses (Wordle would be lost)")

        line = input("\nEnter guess + pattern OR q: ").strip().lower()
        if line == "q":
            return
        parts = line.split()
        if len(parts) != 2:
            print("Format: guess pattern")
            continue
        guess, pat = parts
        if len(guess) != word_len or len(pat) != word_len or any(c not in "012" for c in pat):
            print("Invalid input.")
            continue
        if guess not in allowed_words:
            print("Guess not in dictionary.")
            continue

        guess_count += 1
        if pat == "2" * word_len:
            print("Solved!")
            return
        cands = refine_candidates(cands, guess, pat)
        if not cands:
            print("No candidates remain; something inconsistent.")
            return

def main():
    dictionary_path = "dictionary.txt"
    entropy_cache = "first_guess_entropies.txt"
    max_guesses = 6
    n_jobs = os.cpu_count() or 1

    allowed = load_words(dictionary_path)
    mode_manual_assist(allowed, max_guesses, entropy_cache, n_jobs)

if __name__ == "__main__":
    main()
