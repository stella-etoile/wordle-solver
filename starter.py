#!/usr/bin/env python3
import argparse
import math
from collections import Counter
import os
import multiprocessing as mp
import csv
import time
from datetime import datetime
import json

# ----------------------------
# Globals for multiprocessing
# ----------------------------
_ENTROPY_ANSWERS = None

# For joint entropy with RAM cache
_PATTERNS = None          # list[bytearray], shape: [num_candidates][num_answers], values 0..242
_PAT_BASE = 243           # 3^word_length for 5-letter wordle = 243
_NUM_ANSWERS = None


def tfmt(sec):
    if sec < 1:
        return f"{sec*1000:.0f}ms"
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}m{s:.0f}s"


def load_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.append(w)
    if not words:
        raise ValueError(f"No words loaded from {path}")
    length = len(words[0])
    for w in words:
        if len(w) != length:
            raise ValueError("All words in dictionary must have same length")
    return words


# ----------------------------
# Pattern computation
# ----------------------------
def pattern_for(secret, guess):
    """Returns '0/1/2' string like '20110' (kept for compatibility)."""
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


def pattern_id_for(secret, guess):
    """
    Returns pattern encoded as base-3 integer in [0, 3^n).
    For n=5, range is 0..242. This is much faster to count than strings.
    """
    n = len(secret)
    res = [0] * n
    secret_chars = list(secret)
    guess_chars = list(guess)

    # greens
    for i in range(n):
        if guess_chars[i] == secret_chars[i]:
            res[i] = 2
            secret_chars[i] = None
            guess_chars[i] = None

    # yellows
    for i in range(n):
        if guess_chars[i] is not None:
            ch = guess_chars[i]
            # linear search over 5 letters is fine
            if ch in secret_chars:
                res[i] = 1
                secret_chars[secret_chars.index(ch)] = None

    # base-3 encode (left-to-right)
    x = 0
    for d in res:
        x = x * 3 + d
    return x


# ----------------------------
# Entropy scoring
# ----------------------------
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


def joint_entropy(sequence, answers):
    # Slow path (kept for eval mode and fallback)
    total = len(answers)
    counts = Counter()
    for ans in answers:
        key = tuple(pattern_for(ans, g) for g in sequence)
        counts[key] += 1
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H


# ----------------------------
# Fast joint entropy using RAM cache (pattern_id table)
# ----------------------------
def precompute_candidate_patterns(answers, candidate_words):
    """
    Build patterns table for (candidate_words x answers) where each entry is a uint8 (0..242).
    Memory: len(candidate_words) * len(answers) bytes (≈ 9.6MB for 600x15921).
    """
    A = len(answers)
    patterns = [bytearray(A) for _ in range(len(candidate_words))]
    t0 = time.time()
    for wi, w in enumerate(candidate_words):
        row = patterns[wi]
        for ai, ans in enumerate(answers):
            row[ai] = pattern_id_for(ans, w)
        # lightweight progress
        if (wi + 1) % 25 == 0 or (wi + 1) == len(candidate_words):
            elapsed = time.time() - t0
            eta = (elapsed / (wi + 1)) * (len(candidate_words) - (wi + 1)) if wi + 1 else 0
            print(
                f"\r[cache] {wi+1}/{len(candidate_words)} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}",
                end="",
                flush=True,
            )
    print()
    return patterns


def joint_entropy_cached(seq_indices):
    """
    seq_indices: tuple/list of candidate indices
    Uses global _PATTERNS (candidate x answers) with uint8 values (0..242).
    Encodes the multi-guess pattern key as a single integer in base 243 to avoid tuple allocation.
    """
    patterns = _PATTERNS
    total = _NUM_ANSWERS
    L = len(seq_indices)

    counts = {}
    # Iterate answers, compute base-243 key
    for ai in range(total):
        k = 0
        for idx in seq_indices:
            k = k * _PAT_BASE + patterns[idx][ai]
        counts[k] = counts.get(k, 0) + 1

    H = 0.0
    inv_total = 1.0 / total
    for c in counts.values():
        p = c * inv_total
        H -= p * math.log2(p)
    return H


def _entropy_init(answers):
    global _ENTROPY_ANSWERS
    _ENTROPY_ANSWERS = answers


def _entropy_worker(w):
    H = entropy_for_guess(w, _ENTROPY_ANSWERS)
    return w, H


def compute_single_entropies(guesses, answers, n_jobs, label="[H1]"):
    scores = []
    total = len(guesses)
    if total == 0:
        return scores
    start = time.time()
    if n_jobs <= 1:
        for i, w in enumerate(guesses, 1):
            H = entropy_for_guess(w, answers)
            scores.append((H, w))
            if i % 50 == 0 or i == total:
                elapsed = time.time() - start
                eta = (elapsed / i) * (total - i) if i > 0 else 0
                print(f"\r{label} {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()
    else:
        with mp.Pool(n_jobs, initializer=_entropy_init, initargs=(answers,)) as pool:
            for i, (w, H) in enumerate(pool.imap_unordered(_entropy_worker, guesses), 1):
                scores.append((H, w))
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - start
                    eta = (elapsed / i) * (total - i) if i > 0 else 0
                    print(f"\r{label} {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()
    scores.sort(reverse=True)
    return scores


def load_precomputed_entropies(path, allowed):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            w, h = parts[0], parts[1]
            try:
                v = float(h)
            except ValueError:
                continue
            mapping[w.lower()] = v
    scores = []
    missing = []
    for w in allowed:
        if w in mapping:
            scores.append((mapping[w], w))
        else:
            missing.append(w)
    scores.sort(reverse=True)
    return scores, missing


def get_single_entropies(allowed, answers, n_jobs, first_entropy_file=None):
    if first_entropy_file is None:
        t0 = time.time()
        print("Computing single-word entropies...")
        single = compute_single_entropies(allowed, answers, n_jobs, label="[H1]")
        print(f"Single-word entropy done in {tfmt(time.time() - t0)}")
        ent_dict = {w: H for H, w in single}
        return single, ent_dict
    t0 = time.time()
    print(f"Loading precomputed entropies from {first_entropy_file}...")
    pre_scores, missing = load_precomputed_entropies(first_entropy_file, allowed)
    ent_dict = {w: H for H, w in pre_scores}
    print(f"Loaded {len(pre_scores)} entropies, {len(missing)} missing")
    if missing:
        print("Computing missing single-word entropies...")
        extra = compute_single_entropies(missing, answers, n_jobs, label="[H1-miss]")
        for H, w in extra:
            ent_dict[w] = H
        all_scores = pre_scores + extra
        all_scores.sort(reverse=True)
        print(f"Single-word entropy (precomputed+missing) done in {tfmt(time.time() - t0)}")
        return all_scores, ent_dict
    print(f"Single-word entropy (precomputed only) done in {tfmt(time.time() - t0)}")
    all_scores = pre_scores
    ent_dict = {w: H for H, w in all_scores}
    return all_scores, ent_dict


def _joint_init_cached(patterns, num_answers):
    global _PATTERNS, _NUM_ANSWERS
    _PATTERNS = patterns
    _NUM_ANSWERS = num_answers


def _joint_worker_cached(args):
    base_idx_seq, new_idx = args
    seq = tuple(base_idx_seq) + (new_idx,)
    H = joint_entropy_cached(seq)
    return H, seq


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_level_csv(out_dir, L, level, meta=None):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"L{L:02d}_beam.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "entropy_bits", "sequence"])
        for i, (H, seq) in enumerate(level, 1):
            w.writerow([i, f"{H:.6f}", " ".join(seq)])
    if meta is not None:
        mpath = os.path.join(out_dir, f"L{L:02d}_meta.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    return path


def write_run_meta(out_dir, meta):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return path


def write_candidates(out_dir, candidate_words, ent_dict):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "candidates.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "H1_bits"])
        for i, word in enumerate(candidate_words, 1):
            w.writerow([i, word, f"{ent_dict.get(word, float('nan')):.6f}"])
    return path


def search_best_starters(
    allowed_words,
    answers,
    max_len,
    top_m_words,
    beam_width,
    top_print,
    n_jobs,
    first_entropy_file,
    out_dir=None,
    out_prefix=None,
    save_top_print_only=False,
    use_ram_cache=True,
):
    run_start = time.time()
    single_entropies, ent_dict = get_single_entropies(allowed_words, answers, n_jobs, first_entropy_file)

    candidate_words = [w for _, w in single_entropies[:top_m_words]]
    word_to_cidx = {w: i for i, w in enumerate(candidate_words)}

    # Save setup
    save_dir = None
    if out_dir is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = out_prefix or f"search_M{top_m_words}_B{beam_width}_L{max_len}_{ts}"
        save_dir = os.path.join(out_dir, name)
        ensure_dir(save_dir)
        write_run_meta(
            save_dir,
            {
                "dictionary_size": len(allowed_words),
                "answers_size": len(answers),
                "max_len": max_len,
                "top_m_words": top_m_words,
                "beam_width": beam_width,
                "top_print": top_print,
                "n_jobs": n_jobs,
                "first_entropy_file": first_entropy_file,
                "use_ram_cache": use_ram_cache,
                "started_epoch": int(time.time()),
            },
        )
        write_candidates(save_dir, candidate_words, ent_dict)

    # ----------------------------
    # RAM cache: precompute patterns for candidate words against all answers
    # ----------------------------
    patterns = None
    if use_ram_cache:
        print(f"\nBuilding RAM cache for {len(candidate_words)} candidates x {len(answers)} answers...")
        patterns = precompute_candidate_patterns(answers, candidate_words)
        # set globals for single-process code paths if needed
        global _PATTERNS, _NUM_ANSWERS
        _PATTERNS = patterns
        _NUM_ANSWERS = len(answers)

        # Show approximate RAM footprint (uint8)
        approx_mb = (len(candidate_words) * len(answers)) / (1024 * 1024)
        print(f"[cache] patterns table ≈ {approx_mb:.2f} MB (uint8)\n")

    # Build initial beam (L1)
    level = []
    cand_set = set(candidate_words)
    for H, w in single_entropies:
        if w in cand_set:
            level.append((H, [w]))
    level.sort(reverse=True)
    level = level[:beam_width]

    print("\n=== Best 1-word starters ===")
    for H, seq in level[:top_print]:
        print(f"{' '.join(seq)}: {H:.6f} bits")

    if save_dir is not None:
        to_save = level[:top_print] if save_top_print_only else level
        write_level_csv(
            save_dir,
            1,
            to_save,
            meta={
                "L": 1,
                "beam_saved": len(to_save),
                "beam_width": beam_width,
                "top_m_words": top_m_words,
                "elapsed_sec": round(time.time() - run_start, 6),
            },
        )

    # Iterate L2..max_len
    for L in range(2, max_len + 1):
        print(f"\nSearching best {L}-word starters...")
        tasks = []

        # Use indices when cached; words otherwise
        if use_ram_cache:
            level_idx = [(H, [word_to_cidx[w] for w in seq]) for H, seq in level]
            for _, seq_idx in level_idx:
                used = set(seq_idx)
                base = tuple(seq_idx)
                for new_idx in range(len(candidate_words)):
                    if new_idx not in used:
                        tasks.append((base, new_idx))
        else:
            for _, seq in level:
                used = set(seq)
                base = tuple(seq)
                for w in candidate_words:
                    if w not in used:
                        tasks.append((base, w))

        total = len(tasks)
        if total == 0:
            print("No expansions possible.")
            break

        new_level = []
        start = time.time()

        if use_ram_cache:
            # Cached joint entropy path (multiprocessing supported; patterns copied once per worker)
            if n_jobs <= 1:
                for i, (base_idx, new_idx) in enumerate(tasks, 1):
                    seq_idx = tuple(base_idx) + (new_idx,)
                    H = joint_entropy_cached(seq_idx)
                    seq_words = [candidate_words[j] for j in seq_idx]
                    new_level.append((H, seq_words))
                    if i % 50 == 0 or i == total:
                        elapsed = time.time() - start
                        eta = (elapsed / i) * (total - i) if i > 0 else 0
                        print(f"\r[L{L}] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
                print()
            else:
                with mp.Pool(n_jobs, initializer=_joint_init_cached, initargs=(patterns, len(answers))) as pool:
                    for i, (H, seq_idx) in enumerate(pool.imap_unordered(_joint_worker_cached, tasks), 1):
                        seq_words = [candidate_words[j] for j in seq_idx]
                        new_level.append((H, seq_words))
                        if i % 50 == 0 or i == total:
                            elapsed = time.time() - start
                            eta = (elapsed / i) * (total - i) if i > 0 else 0
                            print(f"\r[L{L}] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
                print()
        else:
            # Old slow path
            if n_jobs <= 1:
                for i, (base, w) in enumerate(tasks, 1):
                    seq_list = list(base) + [w]
                    H = joint_entropy(seq_list, answers)
                    new_level.append((H, seq_list))
                    if i % 20 == 0 or i == total:
                        elapsed = time.time() - start
                        eta = (elapsed / i) * (total - i) if i > 0 else 0
                        print(f"\r[L{L}] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
                print()
            else:
                # keep original mp path for slow mode
                def _joint_init_slow(answers_):
                    global _ENTROPY_ANSWERS
                    _ENTROPY_ANSWERS = answers_

                def _joint_worker_slow(args_):
                    base_seq, new_word = args_
                    seq_ = list(base_seq) + [new_word]
                    return joint_entropy(seq_, _ENTROPY_ANSWERS), tuple(seq_)

                with mp.Pool(n_jobs, initializer=_joint_init_slow, initargs=(answers,)) as pool:
                    for i, (H, seq_tuple) in enumerate(pool.imap_unordered(_joint_worker_slow, tasks), 1):
                        new_level.append((H, list(seq_tuple)))
                        if i % 20 == 0 or i == total:
                            elapsed = time.time() - start
                            eta = (elapsed / i) * (total - i) if i > 0 else 0
                            print(f"\r[L{L}] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
                print()

        new_level.sort(reverse=True)
        level = new_level[:beam_width]

        print(f"\n=== Best {L}-word starters ===")
        for H, seq in level[:top_print]:
            print(f"{' '.join(seq)}: {H:.6f} bits")

        if save_dir is not None:
            to_save = level[:top_print] if save_top_print_only else level
            write_level_csv(
                save_dir,
                L,
                to_save,
                meta={
                    "L": L,
                    "tasks": total,
                    "beam_saved": len(to_save),
                    "beam_width": beam_width,
                    "top_m_words": top_m_words,
                    "elapsed_sec_level": round(time.time() - start, 6),
                    "elapsed_sec_total": round(time.time() - run_start, 6),
                    "use_ram_cache": use_ram_cache,
                },
            )

    if save_dir is not None:
        write_run_meta(
            save_dir,
            {
                "dictionary_size": len(allowed_words),
                "answers_size": len(answers),
                "max_len": max_len,
                "top_m_words": top_m_words,
                "beam_width": beam_width,
                "top_print": top_print,
                "n_jobs": n_jobs,
                "first_entropy_file": first_entropy_file,
                "use_ram_cache": use_ram_cache,
                "finished_epoch": int(time.time()),
                "total_elapsed_sec": round(time.time() - run_start, 6),
            },
        )
        print(f"\nSaved run state to: {save_dir}")


def exhaustive_pair_search(allowed_words, answers, top_m_words, out_csv, n_jobs, pair_dissimilar_only, max_overlap, first_entropy_file):
    # Left as-is (can be upgraded similarly if you want)
    single_entropies, ent_dict = get_single_entropies(allowed_words, answers, n_jobs, first_entropy_file)
    seeds = [w for _, w in single_entropies[:top_m_words]]
    letter_sets = {w: set(w) for w in allowed_words}
    print("\n=== Seed words ===")
    for H, w in single_entropies[:top_m_words]:
        print(f"{w}: {H:.6f} bits")
    tasks = []
    for first in seeds:
        s_first = letter_sets[first]
        for second in allowed_words:
            if second == first:
                continue
            if pair_dissimilar_only:
                overlap = len(s_first & letter_sets[second])
                if overlap > max_overlap:
                    continue
            tasks.append((first, second))
    total = len(tasks)
    print(f"\nTotal pairs: {total}")
    results = []
    start = time.time()
    if n_jobs <= 1:
        for i, (first, second) in enumerate(tasks, 1):
            H = joint_entropy([first, second], answers)
            results.append((H, first, second))
            if i % 50 == 0 or i == total:
                elapsed = time.time() - start
                eta = (elapsed / i) * (total - i) if i > 0 else 0
                print(f"\r[Pairs] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()
    else:
        def _pair_init_slow(answers_):
            global _ENTROPY_ANSWERS
            _ENTROPY_ANSWERS = answers_

        def _pair_worker_slow(args_):
            first_, second_ = args_
            return first_, second_, joint_entropy([first_, second_], _ENTROPY_ANSWERS)

        with mp.Pool(n_jobs, initializer=_pair_init_slow, initargs=(answers,)) as pool:
            for i, (first, second, H) in enumerate(pool.imap_unordered(_pair_worker_slow, tasks), 1):
                results.append((H, first, second))
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - start
                    eta = (elapsed / i) * (total - i) if i > 0 else 0
                    print(f"\r[Pairs] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()

    results.sort(reverse=True)
    print("\n=== Top 20 pairs ===")
    for H, first, second in results[:20]:
        print(f"{first} {second}: {H:.6f} bits")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["first", "second", "joint_entropy", "first_entropy", "second_entropy"])
        for H, first, second in results:
            H1 = ent_dict.get(first, float("nan"))
            H2 = ent_dict.get(second, float("nan"))
            w.writerow([first, second, f"{H:.6f}", f"{H1:.6f}", f"{H2:.6f}"])
    print(f"\nWrote CSV: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dictionary", default="dictionary.txt")
    p.add_argument("--answers", default=None)
    p.add_argument("--mode", choices=["search", "eval", "pairs"], default="search")
    p.add_argument("--starter", nargs="+", default=None)
    p.add_argument("--max-len", type=int, default=3)
    p.add_argument("--top-m-words", type=int, default=50)
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument("--top-print", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=0)
    p.add_argument("--pair-out", default="pairs.csv")
    p.add_argument("--pair-dissimilar-only", action="store_true")
    p.add_argument("--max-overlap", type=int, default=2)
    p.add_argument("--first-entropy-file", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--save-prefix", default=None)
    p.add_argument("--save-top-only", action="store_true")
    p.add_argument("--no-ram-cache", action="store_true", help="Disable RAM cache and use slow joint_entropy()")
    args = p.parse_args()

    allowed = load_words(args.dictionary)
    answers = load_words(args.answers) if args.answers else allowed
    n_jobs = args.n_jobs or (os.cpu_count() or 1)

    if args.mode == "search":
        search_best_starters(
            allowed_words=allowed,
            answers=answers,
            max_len=args.max_len,
            top_m_words=args.top_m_words,
            beam_width=args.beam_width,
            top_print=args.top_print,
            n_jobs=n_jobs,
            first_entropy_file=args.first_entropy_file,
            out_dir=args.save_dir,
            out_prefix=args.save_prefix,
            save_top_print_only=args.save_top_only,
            use_ram_cache=(not args.no_ram_cache),
        )
    elif args.mode == "eval":
        if not args.starter:
            raise SystemExit("Provide --starter word1 word2 ...")
        seq = [w.lower() for w in args.starter]
        for w in seq:
            if w not in allowed:
                raise SystemExit(f"Starter word {w} not in dictionary")
        t0 = time.time()
        H = joint_entropy(seq, answers)  # eval keeps simple, slow but fine for one sequence
        print(f"Starter: {' '.join(seq)}")
        print(f"Entropy: {H:.6f} bits")
        print(f"Time: {tfmt(time.time() - t0)}")
    elif args.mode == "pairs":
        exhaustive_pair_search(
            allowed_words=allowed,
            answers=answers,
            top_m_words=args.top_m_words,
            out_csv=args.pair_out,
            n_jobs=n_jobs,
            pair_dissimilar_only=args.pair_dissimilar_only,
            max_overlap=args.max_overlap,
            first_entropy_file=args.first_entropy_file,
        )


if __name__ == "__main__":
    main()

