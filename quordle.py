import math
from collections import Counter
import os
import multiprocessing

_ENTROPY_STATE = None

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
    if total <= 1:
        return 0.0
    counts = Counter()
    for ans in possible_answers:
        p = pattern_for(ans, guess)
        counts[p] += 1
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H

def _init_entropy_state(state):
    global _ENTROPY_STATE
    _ENTROPY_STATE = state

def _quordle_entropy_worker(guess):
    state = _ENTROPY_STATE
    cands_list = state["cands_list"]
    active = state["active"]
    total_H = 0.0
    for i in range(4):
        if active[i]:
            total_H += entropy_for_guess(guess, cands_list[i])
    return guess, total_H

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

def ensure_first_entropy_cache(path, allowed_words, n_jobs):
    ent = {}
    word_set = set(allowed_words)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
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

    print("Precomputing first-guess entropies (single-board)…")
    total = len(allowed_words)
    n_jobs = n_jobs or (os.cpu_count() or 1)
    last_msg = None
    ent = {}

    with multiprocessing.Pool(
        processes=n_jobs,
        initializer=_init_entropy_state,
        initargs=({"cands_list": [allowed_words], "active": [True]},),
    ) as pool:
        for i, (w, H) in enumerate(pool.imap_unordered(lambda g: (g, entropy_for_guess(g, allowed_words)), allowed_words), 1):
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

def top_quordle_entropy_guesses(
    allowed_words,
    cands_list,
    active,
    k=10,
    restrict_to_candidates=False,
    first_turn_cache=None,
    n_jobs=1,
    progress_label=None,
):
    if restrict_to_candidates:
        pool_words = set()
        for i in range(4):
            if active[i]:
                pool_words.update(cands_list[i])
        pool_words = list(pool_words)
    else:
        pool_words = allowed_words

    if not pool_words:
        return []

    full_first_turn = all(active) and all(len(c) == len(allowed_words) for c in cands_list)
    if full_first_turn and first_turn_cache is not None:
        scores = []
        total = len(pool_words)
        last_msg = None
        for i, w in enumerate(pool_words, 1):
            H1 = first_turn_cache.get(w)
            if H1 is None:
                H1 = entropy_for_guess(w, allowed_words)
            scores.append((4.0 * H1, w))
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
        state = {"cands_list": cands_list, "active": active}
        with multiprocessing.Pool(
            processes=n_jobs,
            initializer=_init_entropy_state,
            initargs=(state,),
        ) as pool:
            for i, (w, H) in enumerate(pool.imap_unordered(_quordle_entropy_worker, pool_words), 1):
                scores.append((H, w))
                last_msg = _print_progress_inline(progress_label, i, total, last_msg)
    else:
        for i, w in enumerate(pool_words, 1):
            H = 0.0
            for b in range(4):
                if active[b]:
                    H += entropy_for_guess(w, cands_list[b])
            scores.append((H, w))
            last_msg = _print_progress_inline(progress_label, i, total, last_msg)

    if last_msg:
        _clear_progress_inline(last_msg)
    scores.sort(reverse=True)
    return scores[:k] if k < len(scores) else scores

def _fmt_board(i):
    return f"[{i+1}]"

def print_quordle_state(cands_list, active, show=40):
    for i in range(4):
        tag = _fmt_board(i)
        if not active[i]:
            print(f"{tag} solved")
            continue
        c = cands_list[i]
        print(f"{tag} possible answers left: {len(c)}")
        if len(c) <= show:
            print(f"{tag} " + " ".join(c))
        else:
            print(f"{tag} " + " ".join(c[:show]))

def mode_quordle_manual_assist(allowed_words, max_guesses, entropy_cache_path, n_jobs):
    word_len = len(allowed_words[0])
    cands_list = [allowed_words[:], allowed_words[:], allowed_words[:], allowed_words[:]]
    active = [True, True, True, True]
    guess_count = 0
    first_ent = None

    while True:
        print()
        print_quordle_state(cands_list, active)

        if not any(active):
            print("\nAll 4 boards solved!")
            return

        print("\nTop high-entropy words (sum across unsolved boards):")
        if guess_count == 0:
            if first_ent is None:
                first_ent = ensure_first_entropy_cache(entropy_cache_path, allowed_words, n_jobs)
            ranking = sorted(((4.0 * H, w) for w, H in first_ent.items()), reverse=True)
            top10 = ranking[:10]
        else:
            top10 = top_quordle_entropy_guesses(
                allowed_words=allowed_words,
                cands_list=cands_list,
                active=active,
                k=10,
                restrict_to_candidates=False,
                first_turn_cache=None,
                n_jobs=n_jobs,
                progress_label="[quordle entropy]",
            )

        for H, w in top10:
            mark = ""
            for b in range(4):
                if active[b] and w in cands_list[b]:
                    mark = "*"
                    break
            print(f"{w}{mark}: {H:.4f}")

        if guess_count >= max_guesses:
            print("\nReached max guesses (Quordle would be lost)")

        line = input("\nEnter: guess p1 p2 p3 p4  (use '-' for already-solved boards) OR q: ").strip().lower()
        if line == "q":
            return

        parts = line.split()
        if len(parts) != 5:
            print("Format: guess p1 p2 p3 p4")
            continue

        guess = parts[0]
        pats = parts[1:]

        if len(guess) != word_len:
            print("Invalid guess length.")
            continue
        if guess not in allowed_words:
            print("Guess not in dictionary.")
            continue

        ok = True
        for p in pats:
            if p == "-":
                continue
            if len(p) != word_len or any(c not in "012" for c in p):
                ok = False
                break
        if not ok:
            print("Invalid patterns. Each must be 0/1/2 of word length, or '-' to skip.")
            continue

        guess_count += 1

        for b in range(4):
            if not active[b]:
                continue
            p = pats[b]
            if p == "-":
                continue
            if p == "2" * word_len:
                active[b] = False
                cands_list[b] = [guess]
                continue
            cands_list[b] = refine_candidates(cands_list[b], guess, p)
            if not cands_list[b]:
                print(f"{_fmt_board(b)} no candidates remain; something inconsistent.")
                return

def main():
    dictionary_path = "dictionary.txt"
    entropy_cache = "first_guess_entropies.txt"
    max_guesses = 9
    n_jobs = os.cpu_count() or 1

    allowed = load_words(dictionary_path)
    mode_quordle_manual_assist(allowed, max_guesses, entropy_cache, n_jobs)

if __name__ == "__main__":
    main()
