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

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

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

def bucket_counts_for_guess(guess, possible_answers):
    counts = Counter()
    for ans in possible_answers:
        counts[pattern_for(ans, guess)] += 1
    return counts

def entropy_from_counts(counts, total):
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H

def entropy_for_guess(guess, possible_answers):
    total = len(possible_answers)
    if total <= 1:
        return 0.0
    counts = bucket_counts_for_guess(guess, possible_answers)
    return entropy_from_counts(counts, total)

def efficiency_for_guess(guess, possible_answers):
    total = len(possible_answers)
    if total <= 1:
        return {
            "H": 0.0,
            "H_norm": 0.0,
            "exp_left": float(total),
            "exp_elim": 0.0,
            "exp_reduction": 0.0,
            "worst_left": total,
            "best_left": total,
            "n_buckets": 1,
        }

    counts = bucket_counts_for_guess(guess, possible_answers)
    H = entropy_from_counts(counts, total)
    exp_left = sum(c * c for c in counts.values()) / total
    exp_elim = total - exp_left
    exp_reduction = 1.0 - (exp_left / total)
    worst_left = max(counts.values()) if counts else total
    best_left = min(counts.values()) if counts else total
    H_norm = H / math.log2(total) if total > 1 else 0.0

    return {
        "H": H,
        "H_norm": H_norm,
        "exp_left": exp_left,
        "exp_elim": exp_elim,
        "exp_reduction": exp_reduction,
        "worst_left": worst_left,
        "best_left": best_left,
        "n_buckets": len(counts),
    }

def multi_efficiency_for_guess(guess, cands_list, active):
    out = {"boards": [], "sum_H": 0.0, "sum_exp_left": 0.0, "sum_exp_elim": 0.0}
    sum_exp_reduction = 0.0
    max_worst = 0
    min_best = None
    sum_buckets = 0
    sum_H_norm = 0.0
    active_count = 0

    for i in range(len(cands_list)):
        if not active[i]:
            out["boards"].append(None)
            continue
        eff = efficiency_for_guess(guess, cands_list[i])
        out["boards"].append(eff)
        out["sum_H"] += eff["H"]
        out["sum_exp_left"] += eff["exp_left"]
        out["sum_exp_elim"] += eff["exp_elim"]
        sum_exp_reduction += eff["exp_reduction"]
        sum_buckets += eff["n_buckets"]
        sum_H_norm += eff["H_norm"]
        active_count += 1
        if eff["worst_left"] > max_worst:
            max_worst = eff["worst_left"]
        if min_best is None or eff["best_left"] < min_best:
            min_best = eff["best_left"]

    out["avg_H_norm"] = (sum_H_norm / active_count) if active_count > 0 else 0.0
    out["avg_exp_reduction"] = (sum_exp_reduction / active_count) if active_count > 0 else 0.0
    out["max_worst_left"] = max_worst
    out["min_best_left"] = min_best if min_best is not None else 0
    out["sum_buckets"] = sum_buckets
    out["active_boards"] = active_count
    return out

def _init_entropy_state(state):
    global _ENTROPY_STATE
    _ENTROPY_STATE = state

def _single_entropy_worker(guess):
    candidates = _ENTROPY_STATE["candidates"]
    return guess, entropy_for_guess(guess, candidates)

def _multi_entropy_worker(guess):
    state = _ENTROPY_STATE
    cands_list = state["cands_list"]
    active = state["active"]
    H = 0.0
    for i in range(len(cands_list)):
        if active[i]:
            H += entropy_for_guess(guess, cands_list[i])
    return guess, H

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

    print("Precomputing first-guess entropies…")
    total = len(allowed_words)
    n_jobs = n_jobs or (os.cpu_count() or 1)
    last_msg = None
    ent = {}

    state = {"candidates": allowed_words}
    with multiprocessing.Pool(
        processes=n_jobs,
        initializer=_init_entropy_state,
        initargs=(state,),
    ) as pool:
        for i, (w, H) in enumerate(pool.imap_unordered(_single_entropy_worker, allowed_words), 1):
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

def top_multi_entropy_guesses(
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
        for i in range(len(cands_list)):
            if active[i]:
                pool_words.update(cands_list[i])
        pool_words = list(pool_words)
    else:
        pool_words = allowed_words

    if not pool_words:
        return []

    n_boards = len(cands_list)
    full_first_turn = all(active) and all(len(c) == len(allowed_words) for c in cands_list)
    if full_first_turn and first_turn_cache is not None:
        scores = []
        total = len(pool_words)
        last_msg = None
        for i, w in enumerate(pool_words, 1):
            H1 = first_turn_cache.get(w)
            if H1 is None:
                H1 = entropy_for_guess(w, allowed_words)
            scores.append((n_boards * H1, w))
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
            for i, (w, H) in enumerate(pool.imap_unordered(_multi_entropy_worker, pool_words), 1):
                scores.append((H, w))
                last_msg = _print_progress_inline(progress_label, i, total, last_msg)
    else:
        for i, w in enumerate(pool_words, 1):
            H = 0.0
            for b in range(len(cands_list)):
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

def print_multi_state(cands_list, active, show=40):
    for i in range(len(cands_list)):
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

def singleton_candidates(cands_list, active):
    word_to_boards = {}
    for i in range(len(cands_list)):
        if not active[i]:
            continue
        c = cands_list[i]
        if len(c) == 1:
            w = c[0]
            if w not in word_to_boards:
                word_to_boards[w] = []
            word_to_boards[w].append(i)
    items = [(w, boards) for w, boards in word_to_boards.items()]
    items.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
    return items

def mode_octordle_manual_assist(allowed_words, max_guesses, entropy_cache_path, n_jobs):
    n_boards = 8
    word_len = len(allowed_words[0])
    cands_list = [allowed_words[:] for _ in range(n_boards)]
    active = [True] * n_boards
    guess_count = 0
    first_ent = None

    while True:
        clear_screen()
        print("=== OCTORDLE MANUAL ASSIST ===")
        print(f"Guesses used: {guess_count}/{max_guesses}")
        print()
        print_multi_state(cands_list, active)

        if not any(active):
            print("\nAll boards solved!")
            return

        singles = singleton_candidates(cands_list, active)
        if singles:
            print("\nGuaranteed answers available (boards with exactly 1 candidate):")
            ranked = []
            for w, boards in singles:
                meff = multi_efficiency_for_guess(w, cands_list, active)
                ranked.append((meff["sum_H"], w, boards, meff))
            ranked.sort(reverse=True, key=lambda x: x[0])
            for Hsum, w, boards, meff in ranked:
                btxt = ",".join(str(i + 1) for i in boards)
                print(
                    f"{w} -> solves [{btxt}]: "
                    f"Hsum={meff['sum_H']:.4f} "
                    f"avgNorm={meff['avg_H_norm']:.3f} "
                    f"EleftSum={meff['sum_exp_left']:.1f} "
                    f"worstMax={meff['max_worst_left']} "
                    f"avgRed={meff['avg_exp_reduction']*100:.1f}% "
                    f"bucketsSum={meff['sum_buckets']}"
                )

        print("\nTop high-entropy words (sum across unsolved boards):")
        if guess_count == 0:
            if first_ent is None:
                first_ent = ensure_first_entropy_cache(entropy_cache_path, allowed_words, n_jobs)
            ranking = sorted(((n_boards * H, w) for w, H in first_ent.items()), reverse=True)
            top10 = ranking[:10]
        else:
            top10 = top_multi_entropy_guesses(
                allowed_words=allowed_words,
                cands_list=cands_list,
                active=active,
                k=10,
                restrict_to_candidates=False,
                first_turn_cache=None,
                n_jobs=n_jobs,
                progress_label="[octordle entropy]",
            )

        for H, w in top10:
            mark = ""
            for b in range(n_boards):
                if active[b] and w in cands_list[b]:
                    mark = "*"
                    break

            meff = multi_efficiency_for_guess(w, cands_list, active)
            print(
                f"{w}{mark}: "
                f"Hsum={meff['sum_H']:.4f} "
                f"avgNorm={meff['avg_H_norm']:.3f} "
                f"EleftSum={meff['sum_exp_left']:.1f} "
                f"worstMax={meff['max_worst_left']} "
                f"avgRed={meff['avg_exp_reduction']*100:.1f}% "
                f"bucketsSum={meff['sum_buckets']}"
            )

        if guess_count >= max_guesses:
            print("\nReached max guesses (Octordle would be lost)")

        line = input(
            f"\nEnter: guess " + " ".join([f"p{i+1}" for i in range(n_boards)]) +
            "  (use '-' for solved boards) OR q: "
        ).strip().lower()
        if line == "q":
            return

        parts = line.split()
        if len(parts) != 1 + n_boards:
            print(f"Format: guess " + " ".join([f"p{i+1}" for i in range(n_boards)]))
            input("Press Enter to continue...")
            continue

        guess = parts[0]
        pats = parts[1:]

        if len(guess) != word_len:
            print("Invalid guess length.")
            input("Press Enter to continue...")
            continue
        if guess not in allowed_words:
            print("Guess not in dictionary.")
            input("Press Enter to continue...")
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
            input("Press Enter to continue...")
            continue

        guess_count += 1

        for b in range(n_boards):
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
                input("Press Enter to exit...")
                return

def main():
    dictionary_path = "dictionary.txt"
    entropy_cache = "first_guess_entropies.txt"
    max_guesses = 13
    n_jobs = os.cpu_count() or 1

    allowed = load_words(dictionary_path)
    mode_octordle_manual_assist(allowed, max_guesses, entropy_cache, n_jobs)

if __name__ == "__main__":
    main()
