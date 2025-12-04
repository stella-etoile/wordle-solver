import argparse
import math
import random
from collections import Counter
import os
import multiprocessing

_ENTROPY_CANDIDATES = None
_BM_ALLOWED = None
_BM_STARTERS = None
_BM_MAX_GUESSES = None
_BM_FIRST_ENT = None

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
        with multiprocessing.Pool(processes=n_jobs, initializer=_entropy_init, initargs=(candidates,)) as pool:
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
    with multiprocessing.Pool(n_jobs, initializer=_entropy_init, initargs=(allowed_words,)) as pool:
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

def play_random_game(secret, allowed_words, starters, max_guesses, game_idx=None):
    word_len = len(secret)
    candidates = allowed_words[:]
    guess_count = 0
    if game_idx is not None:
        print(f"[{game_idx}] {secret}")

    for g in starters:
        guess_count += 1
        pattern = pattern_for(secret, g)
        if game_idx is not None:
            print(f"[{game_idx}] {g}, {pattern}")
        if pattern == "2" * word_len:
            if game_idx is not None:
                print(f"[{game_idx}] {'success' if guess_count <= max_guesses else 'fail'} ({guess_count} guesses)")
            return guess_count
        candidates = refine_candidates(candidates, g, pattern)
        if not candidates:
            if game_idx is not None:
                print(f"[{game_idx}] fail")
            return None

    while True:
        if not candidates:
            if game_idx is not None:
                print(f"[{game_idx}] fail")
            return None

        guess = random.choice(candidates)
        guess_count += 1
        pattern = pattern_for(secret, guess)
        if game_idx is not None:
            print(f"[{game_idx}] {guess}, {pattern}")

        if pattern == "2" * word_len:
            if game_idx is not None:
                print(f"[{game_idx}] {'success' if guess_count <= max_guesses else 'fail'} ({guess_count} guesses)")
            return guess_count

        candidates = refine_candidates(candidates, guess, pattern)

def play_entropy_candidates(secret, allowed_words, starters, max_guesses, first_entropy, n_jobs, game_idx=None):
    word_len = len(secret)
    candidates = allowed_words[:]
    guess_count = 0

    if game_idx is not None:
        print(f"[{game_idx}] {secret}")

    for g in starters:
        guess_count += 1
        pattern = pattern_for(secret, g)
        if game_idx is not None:
            print(f"[{game_idx}] {g}, {pattern}")
        if pattern == "2" * word_len:
            if game_idx is not None:
                print(f"[{game_idx}] {'success' if guess_count <= max_guesses else 'fail'} ({guess_count} guesses)")
            return guess_count
        candidates = refine_candidates(candidates, g, pattern)
        if not candidates:
            if game_idx is not None:
                print(f"[{game_idx}] fail")
            return None

    while True:
        if not candidates:
            if game_idx is not None:
                print(f"[{game_idx}] fail")
            return None

        if guess_count == 0 and len(candidates) == len(allowed_words):
            guess = max(candidates, key=lambda w: first_entropy[w])
        else:
            best = top_entropy_guesses(
                allowed_words,
                candidates,
                k=1,
                restrict_to_candidates=True,
                precomputed=None,
                n_jobs=n_jobs,
                progress_label=f"[{game_idx}] entropy mode2 g{guess_count+1}" if game_idx is not None else None,
            )
            if not best:
                if game_idx is not None:
                    print(f"[{game_idx}] fail")
                return None
            guess = best[0][1]

        guess_count += 1
        pattern = pattern_for(secret, guess)
        if game_idx is not None:
            print(f"[{game_idx}] {guess}, {pattern}")

        if pattern == "2" * word_len:
            if game_idx is not None:
                print(f"[{game_idx}] {'success' if guess_count <= max_guesses else 'fail'} ({guess_count} guesses)")
            return guess_count

        candidates = refine_candidates(candidates, guess, pattern)

def positional_bonus(guess, candidates, yellow_forbidden):
    letters = set(ch for w in candidates for ch in w)
    overlap = sum(ch in letters for ch in guess)
    if overlap == 0:
        return -1e9
    bonus = 0.2 * (overlap / len(guess))
    for i, ch in enumerate(guess):
        if ch in yellow_forbidden and i not in yellow_forbidden[ch]:
            bonus += 0.3
    return bonus

def play_entropy_positional(secret, allowed_words, starters, max_guesses, first_entropy, n_jobs, game_idx=None):
    word_len = len(secret)
    candidates = allowed_words[:]
    guess_count = 0
    yellow_forbidden = {}

    if game_idx is not None:
        print(f"[{game_idx}] {secret}")

    for g in starters:
        guess_count += 1
        pattern = pattern_for(secret, g)
        if game_idx is not None:
            print(f"[{game_idx}] {g}, {pattern}")
        for i, ch in enumerate(g):
            if pattern[i] == "1":
                yellow_forbidden.setdefault(ch, set()).add(i)
        if pattern == "2" * word_len:
            if game_idx is not None:
                print(f"[{game_idx}] {'success' if guess_count <= max_guesses else 'fail'} ({guess_count} guesses)")
            return guess_count
        candidates = refine_candidates(candidates, g, pattern)
        if not candidates:
            if game_idx is not None:
                print(f"[{game_idx}] fail")
            return None

    while True:
        if not candidates:
            if game_idx is not None:
                print(f"[{game_idx}] fail")
            return None

        if guess_count == 0 and len(candidates) == len(allowed_words):
            guess = max(candidates, key=lambda w: first_entropy[w])
        else:
            ent_list = top_entropy_guesses(
                allowed_words,
                candidates,
                k=len(candidates),
                restrict_to_candidates=True,
                precomputed=None,
                n_jobs=n_jobs,
                progress_label=f"[{game_idx}] entropy mode3 g{guess_count+1}" if game_idx is not None else None,
            )
            best_score = None
            best_guess = None
            for H, w in ent_list:
                b = positional_bonus(w, candidates, yellow_forbidden)
                sc = -1e9 if b <= -1e8 else (H + b)
                if best_score is None or sc > best_score:
                    best_score = sc
                    best_guess = w
            guess = best_guess

        guess_count += 1
        pattern = pattern_for(secret, guess)
        if game_idx is not None:
            print(f"[{game_idx}] {guess}, {pattern}")

        for i, ch in enumerate(guess):
            if pattern[i] == "1":
                yellow_forbidden.setdefault(ch, set()).add(i)

        if pattern == "2" * word_len:
            if game_idx is not None:
                print(f"[{game_idx}] {'success' if guess_count <= max_guesses else 'fail'} ({guess_count} guesses)")
            return guess_count

        candidates = refine_candidates(candidates, guess, pattern)

def summarize_results(name, results, max_guesses):
    total = len(results)
    successes = [r for r in results if r is not None and r <= max_guesses]
    fails = total - len(successes)
    print(f"\n{name} results")
    print(f"Games: {total}")
    print(f"Max guesses allowed: {max_guesses}")
    print(f"Successes: {len(successes)}")
    print(f"Fails: {fails}")
    if total > 0:
        print(f"Success rate: {len(successes)/total*100:.2f}%")
    if successes:
        print(f"Average guesses (success only): {sum(successes)/len(successes):.3f}")
    numeric = [r for r in results if r is not None]
    if numeric:
        print(f"Average guesses (all): {sum(numeric)/len(numeric):.3f}")
        dist = Counter(numeric)
        print("Distribution (all guesses):")
        for g in sorted(dist):
            print(f"  {g}: {dist[g]}")

def simulate_random_strategy(allowed_words, starters, max_guesses, n_games):
    if n_games <= len(allowed_words):
        targets = random.sample(allowed_words, n_games)
    else:
        targets = [random.choice(allowed_words) for _ in range(n_games)]
    results = []
    for i, sec in enumerate(targets, 1):
        results.append(play_random_game(sec, allowed_words, starters, max_guesses, i))
    summarize_results("Random strategy", results, max_guesses)

def simulate_entropy_candidates(allowed_words, starters, max_guesses, n_games, entropy_cache_path, n_jobs):
    first_ent = ensure_first_entropy_cache(entropy_cache_path, allowed_words, n_jobs)
    if n_games <= len(allowed_words):
        targets = random.sample(allowed_words, n_games)
    else:
        targets = [random.choice(allowed_words) for _ in range(n_games)]
    results = []
    for i, sec in enumerate(targets, 1):
        results.append(play_entropy_candidates(sec, allowed_words, starters, max_guesses, first_ent, n_jobs, i))
    summarize_results("Entropy (candidates-only)", results, max_guesses)

def simulate_entropy_positional(allowed_words, starters, max_guesses, n_games, entropy_cache_path, n_jobs):
    first_ent = ensure_first_entropy_cache(entropy_cache_path, allowed_words, n_jobs)
    if n_games <= len(allowed_words):
        targets = random.sample(allowed_words, n_games)
    else:
        targets = [random.choice(allowed_words) for _ in range(n_games)]
    results = []
    for i, sec in enumerate(targets, 1):
        results.append(play_entropy_positional(sec, allowed_words, starters, max_guesses, first_ent, n_jobs, i))
    summarize_results("Entropy (positional heuristic)", results, max_guesses)

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

def _benchmark_init(allowed_words, starters, max_guesses, first_ent):
    global _BM_ALLOWED, _BM_STARTERS, _BM_MAX_GUESSES, _BM_FIRST_ENT
    _BM_ALLOWED = allowed_words
    _BM_STARTERS = starters
    _BM_MAX_GUESSES = max_guesses
    _BM_FIRST_ENT = first_ent

def _benchmark_worker(task):
    idx, sec = task
    r1 = play_random_game(sec, _BM_ALLOWED, _BM_STARTERS, _BM_MAX_GUESSES, game_idx=None)
    r2 = play_entropy_candidates(sec, _BM_ALLOWED, _BM_STARTERS, _BM_MAX_GUESSES, _BM_FIRST_ENT, 1, game_idx=None)
    r3 = play_entropy_positional(sec, _BM_ALLOWED, _BM_STARTERS, _BM_MAX_GUESSES, _BM_FIRST_ENT, 1, game_idx=None)
    return idx, sec, r1, r2, r3

def benchmark_all_modes(
    allowed_words,
    starters,
    max_guesses,
    n_games,
    entropy_cache_path,
    n_jobs,
):
    first_ent = ensure_first_entropy_cache(entropy_cache_path, allowed_words, n_jobs)

    if n_games <= len(allowed_words):
        targets = random.sample(allowed_words, n_games)
    else:
        targets = [random.choice(allowed_words) for _ in range(n_games)]

    results_random = []
    results_entropy = []
    results_positional = []

    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    existing = [
        fn for fn in os.listdir(logs_dir)
        if fn.startswith("logs_") and fn.endswith(".csv")
    ]
    nums = []
    for fn in existing:
        core = fn[len("logs_"):-len(".csv")]
        try:
            nums.append(int(core))
        except ValueError:
            continue
    next_num = max(nums) + 1 if nums else 0
    log_path = os.path.join(logs_dir, f"logs_{next_num}.csv")

    tasks = list(enumerate(targets))
    collected = []

    n_jobs = n_jobs or (os.cpu_count() or 1)
    with multiprocessing.Pool(
        processes=n_jobs,
        initializer=_benchmark_init,
        initargs=(allowed_words, starters, max_guesses, first_ent),
    ) as pool:
        for idx, sec, r1, r2, r3 in pool.imap_unordered(_benchmark_worker, tasks):
            s1 = int(r1 is not None and r1 <= max_guesses)
            s2 = int(r2 is not None and r2 <= max_guesses)
            s3 = int(r3 is not None and r3 <= max_guesses)

            g1 = "" if r1 is None else r1
            g2 = "" if r2 is None else r2
            g3 = "" if r3 is None else r3

            print(f"[{idx}] {sec}: rand={g1}({s1}) ent={g2}({s2}) pos={g3}({s3})")

            results_random.append(r1)
            results_entropy.append(r2)
            results_positional.append(r3)

            collected.append((idx, sec, g1, s1, g2, s2, g3, s3))

    collected.sort(key=lambda x: x[0])
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("index,answer,rand,rand_s,ent,ent_s,pos,pos_s\n")
        for idx, sec, g1, s1, g2, s2, g3, s3 in collected:
            f.write(f"{idx},{sec},{g1},{s1},{g2},{s2},{g3},{s3}\n")

    print(f"\nMode 5 log saved to {log_path}")

    summarize_results("Random", results_random, max_guesses)
    summarize_results("Entropy (mode 2)", results_entropy, max_guesses)
    summarize_results("Positional (mode 3)", results_positional, max_guesses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, choices=[0, 1, 2, 3, 5], required=True)
    parser.add_argument("--dictionary", default="dictionary.txt")
    parser.add_argument("--starter", default="starter.txt")
    parser.add_argument("--max-guesses", type=int, default=6)
    parser.add_argument("--first-entropy-cache", default="first_guess_entropies.txt")
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--n-games", type=int, default=10000)
    args = parser.parse_args()

    allowed = load_words(args.dictionary)
    starters = load_starters(args.starter, allowed)
    n_jobs = args.n_jobs or (os.cpu_count() or 1)

    if args.mode == 0:
        print("Mode 0: manual assist")
        mode_manual_assist(allowed, args.max_guesses, args.first_entropy_cache, n_jobs)
    elif args.mode == 1:
        print("Mode 1: random solver")
        simulate_random_strategy(allowed, starters, args.max_guesses, args.n_games)
    elif args.mode == 2:
        print("Mode 2: entropy-only solver")
        simulate_entropy_candidates(allowed, starters, args.max_guesses, args.n_games, args.first_entropy_cache, n_jobs)
    elif args.mode == 3:
        print("Mode 3: positional entropy solver")
        simulate_entropy_positional(allowed, starters, args.max_guesses, args.n_games, args.first_entropy_cache, n_jobs)
    elif args.mode == 5:
        print("Mode 5: benchmark modes 1, 2, 3")
        benchmark_all_modes(
            allowed,
            starters,
            args.max_guesses,
            args.n_games,
            args.first_entropy_cache,
            n_jobs,
        )

if __name__ == "__main__":
    main()
