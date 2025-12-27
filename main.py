# simulate_wordle_quordle_octordle_mp.py
import argparse
import csv
import math
import os
import random
import time
import multiprocessing as mp
from collections import Counter, defaultdict

_ALLOWED_WORDS = None
_WORD_LEN = None

def load_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.append(w)
    if not words:
        raise ValueError(f"No words loaded from {path}")
    n = len(words[0])
    for w in words:
        if len(w) != n:
            raise ValueError("All words in dictionary must have same length")
    return words

def _init_worker(dictionary_path):
    global _ALLOWED_WORDS, _WORD_LEN
    _ALLOWED_WORDS = load_words(dictionary_path)
    _WORD_LEN = len(_ALLOWED_WORDS[0])

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
    if total <= 1:
        return 0.0
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

def total_entropy_multi(guess, cands_list, active):
    H = 0.0
    for i in range(len(cands_list)):
        if active[i]:
            H += entropy_for_guess(guess, cands_list[i])
    return H

def best_entropy_guess_single(candidates):
    best = None
    bestH = -1.0
    for w in candidates:
        H = entropy_for_guess(w, candidates)
        if H > bestH:
            bestH = H
            best = w
    return best, bestH

def best_entropy_guess_multi(allowed_words, cands_list, active):
    best = None
    bestH = -1.0
    for w in allowed_words:
        H = total_entropy_multi(w, cands_list, active)
        if H > bestH:
            bestH = H
            best = w
    return best, bestH

def best_entropy_guess_multi_restrict_to_union(cands_list, active):
    pool = set()
    for i in range(len(cands_list)):
        if active[i]:
            pool.update(cands_list[i])
    pool = list(pool)
    if not pool:
        return None, 0.0
    best = None
    bestH = -1.0
    for w in pool:
        H = total_entropy_multi(w, cands_list, active)
        if H > bestH:
            bestH = H
            best = w
    return best, bestH

def wordle_random_solve(secret, allowed_words, max_guesses, rng):
    cands = allowed_words[:]
    n = len(secret)
    for turn in range(1, max_guesses + 1):
        guess = rng.choice(cands) if cands else rng.choice(allowed_words)
        pat = pattern_for(secret, guess)
        if pat == "2" * n:
            return True, turn
        cands = refine_candidates(cands, guess, pat)
        if not cands:
            return False, turn
    return False, max_guesses

def wordle_entropy_only_solve(secret, allowed_words, max_guesses):
    cands = allowed_words[:]
    n = len(secret)
    for turn in range(1, max_guesses + 1):
        if not cands:
            return False, turn
        if len(cands) == 1:
            guess = cands[0]
        else:
            guess, _ = best_entropy_guess_single(cands)
        pat = pattern_for(secret, guess)
        if pat == "2" * n:
            return True, turn
        cands = refine_candidates(cands, guess, pat)
    return False, max_guesses

def wordle_positional_entropy_solve(secret, allowed_words, max_guesses):
    cands = allowed_words[:]
    n = len(secret)
    yellow_forbidden = defaultdict(set)

    for turn in range(1, max_guesses + 1):
        if not cands:
            return False, turn

        if len(cands) == 1:
            guess = cands[0]
        else:
            best = None
            bestScore = -1e18
            for w in cands:
                H = entropy_for_guess(w, cands)
                bad = 0
                good = 0
                for i, ch in enumerate(w):
                    if ch in yellow_forbidden:
                        if i in yellow_forbidden[ch]:
                            bad += 1
                        else:
                            good += 1
                score = H + 0.03 * good - 0.10 * bad
                if score > bestScore:
                    bestScore = score
                    best = w
            guess = best

        pat = pattern_for(secret, guess)
        if pat == "2" * n:
            return True, turn

        for i, (ch, p) in enumerate(zip(guess, pat)):
            if p == "1":
                yellow_forbidden[ch].add(i)

        cands = refine_candidates(cands, guess, pat)

    return False, max_guesses

def multi_game_solve(secrets, allowed_words, max_guesses, n_entropy_first, restrict_guess_pool=True):
    n_boards = len(secrets)
    word_len = len(secrets[0])
    cands_list = [allowed_words[:] for _ in range(n_boards)]
    active = [True] * n_boards
    solved_turn = [None] * n_boards

    for turn in range(1, max_guesses + 1):
        if not any(active):
            break

        if turn <= n_entropy_first:
            if restrict_guess_pool:
                guess, _ = best_entropy_guess_multi_restrict_to_union(cands_list, active)
                if guess is None:
                    guess, _ = best_entropy_guess_multi(allowed_words, cands_list, active)
            else:
                guess, _ = best_entropy_guess_multi(allowed_words, cands_list, active)
        else:
            singleton_idxs = [i for i in range(n_boards) if active[i] and len(cands_list[i]) == 1]
            if singleton_idxs:
                best_i = None
                bestH = -1.0
                for i in singleton_idxs:
                    g = cands_list[i][0]
                    H = total_entropy_multi(g, cands_list, active)
                    if H > bestH:
                        bestH = H
                        best_i = i
                guess = cands_list[best_i][0]
            else:
                if restrict_guess_pool:
                    guess, _ = best_entropy_guess_multi_restrict_to_union(cands_list, active)
                    if guess is None:
                        guess, _ = best_entropy_guess_multi(allowed_words, cands_list, active)
                else:
                    guess, _ = best_entropy_guess_multi(allowed_words, cands_list, active)

        for b in range(n_boards):
            if not active[b]:
                continue
            pat = pattern_for(secrets[b], guess)
            if pat == "2" * word_len:
                active[b] = False
                solved_turn[b] = turn
                cands_list[b] = [guess]
            else:
                cands_list[b] = refine_candidates(cands_list[b], guess, pat)
                if not cands_list[b]:
                    active[b] = False
                    solved_turn[b] = None

    solved = all(t is not None for t in solved_turn)
    turns_taken = max((t for t in solved_turn if t is not None), default=max_guesses)
    score_sum = sum((t if t is not None else max_guesses) for t in solved_turn)
    return solved, min(turns_taken, max_guesses), score_sum, solved_turn

def fmt_eta(seconds):
    if seconds < 0:
        seconds = 0
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def simulate_one_trial(args):
    trial_idx, base_seed = args
    allowed = _ALLOWED_WORDS
    rng = random.Random((base_seed * 1_000_003) ^ (trial_idx * 9_999_991))

    targets = rng.sample(allowed, 8)
    wordle_targets = targets
    quordle1 = targets[0:4]
    quordle2 = targets[4:8]
    octordle = targets

    rows = []

    for i, secret in enumerate(wordle_targets, 1):
        solved, turns = wordle_random_solve(secret, allowed, 6, rng)
        rows.append([
            trial_idx, "wordle", "random", "", f"wordle_{i}", 6,
            int(solved), turns, turns if solved else 6,
            secret, str(turns) if solved else ""
        ])

    for i, secret in enumerate(wordle_targets, 1):
        solved, turns = wordle_entropy_only_solve(secret, allowed, 6)
        rows.append([
            trial_idx, "wordle", "entropy_only", "", f"wordle_{i}", 6,
            int(solved), turns, turns if solved else 6,
            secret, str(turns) if solved else ""
        ])

    for i, secret in enumerate(wordle_targets, 1):
        solved, turns = wordle_positional_entropy_solve(secret, allowed, 6)
        rows.append([
            trial_idx, "wordle", "positional_entropy", "", f"wordle_{i}", 6,
            int(solved), turns, turns if solved else 6,
            secret, str(turns) if solved else ""
        ])

    for n in (1, 2, 3, 4):
        solved, turns_taken, score_sum, solved_turns = multi_game_solve(
            quordle1, allowed, 9, n_entropy_first=n, restrict_guess_pool=True
        )
        rows.append([
            trial_idx, "quordle", "entropy_then_lockin", n, "quordle_1", 9,
            int(solved), turns_taken, score_sum,
            " ".join(quordle1),
            " ".join("" if x is None else str(x) for x in solved_turns),
        ])

        solved, turns_taken, score_sum, solved_turns = multi_game_solve(
            quordle2, allowed, 9, n_entropy_first=n, restrict_guess_pool=True
        )
        rows.append([
            trial_idx, "quordle", "entropy_then_lockin", n, "quordle_2", 9,
            int(solved), turns_taken, score_sum,
            " ".join(quordle2),
            " ".join("" if x is None else str(x) for x in solved_turns),
        ])

    for n in (1, 2, 3, 4):
        solved, turns_taken, score_sum, solved_turns = multi_game_solve(
            octordle, allowed, 13, n_entropy_first=n, restrict_guess_pool=True
        )
        rows.append([
            trial_idx, "octordle", "entropy_then_lockin", n, "octordle_1", 13,
            int(solved), turns_taken, score_sum,
            " ".join(octordle),
            " ".join("" if x is None else str(x) for x in solved_turns),
        ])

    trial_done_msg = f"[trial {trial_idx}] done"
    return trial_idx, rows, trial_done_msg

def run_simulation_mp(dictionary_path, trials, seed, outdir, n_procs, chunksize, verbose):
    ensure_dir(outdir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(outdir, f"sim_{ts}_trials{trials}_seed{seed}_mp{n_procs}.csv")

    header = [
        "trial",
        "game_type",
        "strategy",
        "n_entropy_first",
        "game_id",
        "max_guesses",
        "solved",
        "turns_taken",
        "score_sum",
        "targets",
        "targets_solved_turns",
    ]

    t0 = time.time()
    done = 0
    total = trials
    pending = [(i, seed) for i in range(1, trials + 1)]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(header)

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_procs,
            initializer=_init_worker,
            initargs=(dictionary_path,),
        ) as pool:
            for trial_idx, rows, trial_done_msg in pool.imap_unordered(
                simulate_one_trial, pending, chunksize=chunksize
            ):
                for r in rows:
                    wcsv.writerow(r)

                done += 1
                elapsed = time.time() - t0
                rate = elapsed / done if done > 0 else 0.0
                eta = rate * (total - done)

                msg = (
                    f"\r[trials] {done:4d}/{total}  "
                    f"{int(done*100/total):3d}%  "
                    f"elapsed {fmt_eta(elapsed)}  "
                    f"eta {fmt_eta(eta)}"
                )
                print(msg, end="", flush=True)

                if verbose:
                    print()
                    print(trial_done_msg)

    print()
    print(f"Saved: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dictionary", default="dictionary.txt")
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="sim_logs")
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    n_procs = args.procs if args.procs and args.procs > 0 else (os.cpu_count() or 1)
    run_simulation_mp(
        dictionary_path=args.dictionary,
        trials=args.trials,
        seed=args.seed,
        outdir=args.outdir,
        n_procs=n_procs,
        chunksize=max(1, args.chunksize),
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
