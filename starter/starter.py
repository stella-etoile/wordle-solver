#!/usr/bin/env python3
import argparse
import math
from collections import Counter
import os
import multiprocessing as mp
import csv
import time

_ENTROPY_ANSWERS = None
_JOINT_ANSWERS = None

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

def _joint_init(answers):
    global _JOINT_ANSWERS
    _JOINT_ANSWERS = answers

def _joint_worker(args):
    base_seq, new_word = args
    seq = list(base_seq) + [new_word]
    H = joint_entropy(seq, _JOINT_ANSWERS)
    return H, tuple(seq)

def search_best_starters(
    allowed_words,
    answers,
    max_len,
    top_m_words,
    beam_width,
    top_print,
    n_jobs,
    first_entropy_file,
):
    single_entropies, ent_dict = get_single_entropies(allowed_words, answers, n_jobs, first_entropy_file)
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
            with mp.Pool(n_jobs, initializer=_joint_init, initargs=(answers,)) as pool:
                for i, (H, seq_tuple) in enumerate(pool.imap_unordered(_joint_worker, tasks), 1):
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

def _pair_init(answers):
    global _JOINT_ANSWERS
    _JOINT_ANSWERS = answers

def _pair_worker(args):
    first, second = args
    H = joint_entropy([first, second], _JOINT_ANSWERS)
    return first, second, H

def exhaustive_pair_search(allowed_words, answers, top_m_words, out_csv, n_jobs, pair_dissimilar_only, max_overlap, first_entropy_file):
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
        with mp.Pool(n_jobs, initializer=_pair_init, initargs=(answers,)) as pool:
            for i, (first, second, H) in enumerate(pool.imap_unordered(_pair_worker, tasks), 1):
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
        )
    elif args.mode == "eval":
        if not args.starter:
            raise SystemExit("Provide --starter word1 word2 ...")
        seq = [w.lower() for w in args.starter]
        for w in seq:
            if w not in allowed:
                raise SystemExit(f"Starter word {w} not in dictionary")
        t0 = time.time()
        H = joint_entropy(seq, answers)
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
