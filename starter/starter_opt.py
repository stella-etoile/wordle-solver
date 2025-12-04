#!/usr/bin/env python3
import argparse
import math
from collections import Counter
import os
import multiprocessing as mp
import csv
import time
import numpy as np

_PATTERNS = None
_GUESS_INDEX = None
_NUM_PATTERNS = None
_ALLOWED_ARR = None
_ANSWERS_ARR = None
_PAT_BASE = None

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

def pattern_code_from_arrays(secret_arr, guess_arr, base):
    n = len(secret_arr)
    greens = [0] * n
    freq = [0] * 26
    for i in range(n):
        if guess_arr[i] == secret_arr[i]:
            greens[i] = 2
        else:
            freq[secret_arr[i]] += 1
    digits = [0] * n
    for i in range(n):
        if greens[i] == 2:
            digits[i] = 2
        else:
            g = guess_arr[i]
            if freq[g] > 0:
                digits[i] = 1
                freq[g] -= 1
            else:
                digits[i] = 0
    code = 0
    mul = 1
    for i in range(n):
        code += digits[i] * mul
        mul *= 3
    return code

def _pat_init(allowed_arr, answers_arr, base):
    global _ALLOWED_ARR, _ANSWERS_ARR, _PAT_BASE
    _ALLOWED_ARR = allowed_arr
    _ANSWERS_ARR = answers_arr
    _PAT_BASE = base

def _pat_worker(gi):
    g_arr = _ALLOWED_ARR[gi]
    n_answers = _ANSWERS_ARR.shape[0]
    row = np.empty(n_answers, dtype=np.int16)
    for ai in range(n_answers):
        a_arr = _ANSWERS_ARR[ai]
        row[ai] = pattern_code_from_arrays(a_arr, g_arr, _PAT_BASE)
    return gi, row

def build_pattern_matrix(allowed, answers, patterns_file, n_jobs):
    n_guesses = len(allowed)
    n_answers = len(answers)
    if n_guesses == 0 or n_answers == 0:
        raise ValueError("Empty word lists")
    L = len(allowed[0])
    base = 3 ** L
    print(f"Building pattern matrix for {n_guesses} guesses x {n_answers} answers (L={L})")
    allowed_arr = np.array([[ord(c) - 97 for c in w] for w in allowed], dtype=np.int8)
    answers_arr = np.array([[ord(c) - 97 for c in w] for w in answers], dtype=np.int8)
    mat = np.empty((n_guesses, n_answers), dtype=np.int16)
    start = time.time()
    if n_jobs <= 1:
        for gi in range(n_guesses):
            g_arr = allowed_arr[gi]
            row = mat[gi]
            for ai in range(n_answers):
                a_arr = answers_arr[ai]
                row[ai] = pattern_code_from_arrays(a_arr, g_arr, base)
            if (gi + 1) % 10 == 0 or gi + 1 == n_guesses:
                elapsed = time.time() - start
                eta = (elapsed / (gi + 1)) * (n_guesses - (gi + 1))
                print(f"\r[PAT] {gi+1}/{n_guesses} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()
    else:
        indices = list(range(n_guesses))
        with mp.Pool(n_jobs, initializer=_pat_init, initargs=(allowed_arr, answers_arr, base)) as pool:
            for i, (gi, row) in enumerate(pool.imap_unordered(_pat_worker, indices), 1):
                mat[gi] = row
                if i % 10 == 0 or i == n_guesses:
                    elapsed = time.time() - start
                    eta = (elapsed / i) * (n_guesses - i)
                    print(f"\r[PAT] {i}/{n_guesses} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()
    np.save(patterns_file, mat)
    print(f"Saved pattern matrix to {patterns_file}")
    return mat

def load_or_build_patterns(allowed, answers, patterns_file, n_jobs):
    if patterns_file is not None and os.path.exists(patterns_file):
        mat = np.load(patterns_file)
        if mat.shape != (len(allowed), len(answers)):
            raise ValueError("Existing pattern matrix shape does not match current word lists")
        L = len(allowed[0])
        num_patterns = 3 ** L
        return mat, num_patterns
    if patterns_file is None:
        patterns_file = "patterns.npy"
    mat = build_pattern_matrix(allowed, answers, patterns_file, n_jobs)
    L = len(allowed[0])
    num_patterns = 3 ** L
    return mat, num_patterns

def entropy_from_row(row, num_patterns):
    total = row.size
    counts = np.bincount(row, minlength=num_patterns).astype(np.float64)
    p = counts / total
    mask = counts > 0
    p = p[mask]
    H = -np.sum(p * np.log2(p))
    return float(H)

def joint_entropy_from_rows(rows, num_patterns):
    base = num_patterns
    keys = rows[0].astype(np.int64)
    mul = base
    for r in rows[1:]:
        keys = keys + mul * r.astype(np.int64)
        mul *= base
    unique, counts = np.unique(keys, return_counts=True)
    total = counts.sum()
    p = counts.astype(np.float64) / total
    H = -np.sum(p * np.log2(p))
    return float(H)

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

def _entropy_matrix_init(patterns, guess_index, num_patterns):
    global _PATTERNS, _GUESS_INDEX, _NUM_PATTERNS
    _PATTERNS = patterns
    _GUESS_INDEX = guess_index
    _NUM_PATTERNS = num_patterns

def _entropy_matrix_worker(w):
    gi = _GUESS_INDEX[w]
    row = _PATTERNS[gi]
    H = entropy_from_row(row, _NUM_PATTERNS)
    return w, H

def compute_single_entropies_from_matrix(allowed, answers, patterns, num_patterns, n_jobs, first_entropy_file=None):
    guess_index = {w: i for i, w in enumerate(allowed)}
    ent_dict = {}
    if first_entropy_file is not None:
        print(f"Loading precomputed entropies from {first_entropy_file}...")
        pre_scores, missing = load_precomputed_entropies(first_entropy_file, allowed)
        for H, w in pre_scores:
            ent_dict[w] = H
        print(f"Loaded {len(pre_scores)} entropies, {len(missing)} missing")
        missing_list = missing
    else:
        pre_scores = []
        missing_list = list(allowed)
    if missing_list:
        print("Computing missing single-word entropies from pattern matrix...")
        total = len(missing_list)
        start = time.time()
        if n_jobs <= 1:
            for i, w in enumerate(missing_list, 1):
                gi = guess_index[w]
                row = patterns[gi]
                H = entropy_from_row(row, num_patterns)
                ent_dict[w] = H
                if i % 50 == 0 or i == total:
                    elapsed = time.time() - start
                    eta = (elapsed / i) * (total - i) if i > 0 else 0
                    print(f"\r[H1] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
            print()
        else:
            with mp.Pool(n_jobs, initializer=_entropy_matrix_init, initargs=(patterns, guess_index, num_patterns)) as pool:
                for i, (w, H) in enumerate(pool.imap_unordered(_entropy_matrix_worker, missing_list), 1):
                    ent_dict[w] = H
                    if i % 50 == 0 or i == total:
                        elapsed = time.time() - start
                        eta = (elapsed / i) * (total - i) if i > 0 else 0
                        print(f"\r[H1] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
            print()
    all_scores = [(ent_dict[w], w) for w in allowed]
    all_scores.sort(reverse=True)
    return all_scores, ent_dict, guess_index

def _joint_seq_init(patterns, guess_index, num_patterns):
    global _PATTERNS, _GUESS_INDEX, _NUM_PATTERNS
    _PATTERNS = patterns
    _GUESS_INDEX = guess_index
    _NUM_PATTERNS = num_patterns

def _joint_seq_worker(seq):
    rows = [_PATTERNS[_GUESS_INDEX[w]] for w in seq]
    H = joint_entropy_from_rows(rows, _NUM_PATTERNS)
    return H, seq

def _joint_pair_worker(pair):
    first, second = pair
    rows = [_PATTERNS[_GUESS_INDEX[first]], _PATTERNS[_GUESS_INDEX[second]]]
    H = joint_entropy_from_rows(rows, _NUM_PATTERNS)
    return H, first, second

def joint_entropy_sequence_words(seq, patterns, guess_index, num_patterns):
    rows = [patterns[guess_index[w]] for w in seq]
    return joint_entropy_from_rows(rows, num_patterns)

def search_best_starters(
    allowed_words,
    answers,
    max_len,
    top_m_words,
    beam_width,
    top_print,
    n_jobs,
    first_entropy_file,
    patterns,
    num_patterns,
    guess_index,
):
    single_entropies, ent_dict, guess_index = compute_single_entropies_from_matrix(
        allowed_words, answers, patterns, num_patterns, n_jobs, first_entropy_file
    )
    candidate_words = [w for _, w in single_entropies[:top_m_words]]
    level = []
    for H, w in single_entropies:
        if w in candidate_words:
            level.append((H, [w]))
    level.sort(reverse=True)
    level = level[:beam_width]
    print("\n=== Best 1-word starters ===")
    for H, seq in level[:top_print]:
        print(f"{' '.join(seq)}: {H:.6f} bits")
    for L in range(2, max_len + 1):
        print(f"\nSearching best {L}-word starters...")
        tasks = []
        for H_prev, seq in level:
            used = set(seq)
            base = list(seq)
            for w in candidate_words:
                if w not in used:
                    tasks.append(base + [w])
        total = len(tasks)
        if total == 0:
            print("No expansions possible.")
            break
        new_level = []
        start = time.time()
        if n_jobs <= 1:
            for i, seq in enumerate(tasks, 1):
                H = joint_entropy_sequence_words(seq, patterns, guess_index, num_patterns)
                new_level.append((H, seq))
                if i % 20 == 0 or i == total:
                    elapsed = time.time() - start
                    eta = (elapsed / i) * (total - i) if i > 0 else 0
                    print(f"\r[L{L}] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
            print()
        else:
            with mp.Pool(n_jobs, initializer=_joint_seq_init, initargs=(patterns, guess_index, num_patterns)) as pool:
                for i, (H, seq) in enumerate(pool.imap_unordered(_joint_seq_worker, tasks), 1):
                    new_level.append((H, seq))
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

def exhaustive_pair_search(
    allowed_words,
    answers,
    top_m_words,
    out_csv,
    n_jobs,
    pair_dissimilar_only,
    max_overlap,
    first_entropy_file,
    patterns,
    num_patterns,
    guess_index,
):
    single_entropies, ent_dict, guess_index = compute_single_entropies_from_matrix(
        allowed_words, answers, patterns, num_patterns, n_jobs, first_entropy_file
    )
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
            H = joint_entropy_sequence_words([first, second], patterns, guess_index, num_patterns)
            results.append((H, first, second))
            if i % 50 == 0 or i == total:
                elapsed = time.time() - start
                eta = (elapsed / i) * (total - i) if i > 0 else 0
                print(f"\r[Pairs] {i}/{total} | elapsed {tfmt(elapsed)} | ETA {tfmt(eta)}", end="", flush=True)
        print()
    else:
        with mp.Pool(n_jobs, initializer=_joint_seq_init, initargs=(patterns, guess_index, num_patterns)) as pool:
            for i, (H, first, second) in enumerate(pool.imap_unordered(_joint_pair_worker, tasks), 1):
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
    p.add_argument("--patterns-file", default="patterns.npy")
    args = p.parse_args()
    allowed = load_words(args.dictionary)
    answers = load_words(args.answers) if args.answers else allowed
    n_jobs = args.n_jobs or (os.cpu_count() or 1)
    patterns, num_patterns = load_or_build_patterns(allowed, answers, args.patterns_file, n_jobs)
    guess_index = {w: i for i, w in enumerate(allowed)}
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
            patterns=patterns,
            num_patterns=num_patterns,
            guess_index=guess_index,
        )
    elif args.mode == "eval":
        if not args.starter:
            raise SystemExit("Provide --starter word1 word2 ...")
        seq = [w.lower() for w in args.starter]
        for w in seq:
            if w not in guess_index:
                raise SystemExit(f"Starter word {w} not in dictionary")
        t0 = time.time()
        H = joint_entropy_sequence_words(seq, patterns, guess_index, num_patterns)
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
            patterns=patterns,
            num_patterns=num_patterns,
            guess_index=guess_index,
        )

if __name__ == "__main__":
    main()
