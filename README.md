# Wordle / Quordle / Octordle Entropy Solver

A small command-line helper that suggests strong guesses using **Shannon entropy** (information gain).  
Works in three modes/scripts:

- `wordle.py` (1 board, 6 guesses)
- `quordle.py` (4 boards, 8 guesses)
- `octordle.py` (8 boards, 12 guesses)

You play in your browser/app, then copy the feedback patterns into the CLI to narrow the candidate word list and get new high-entropy suggestions.

---

## Files

- `dictionary.txt`  
  Your word list (one word per line, all same length). This is both the **allowed guesses** and the **possible answers** list.

- `first_guess_entropies.txt`  
  Auto-generated cache of first-turn entropies (speeds up the very first suggestion step).

- `wordle.py`  
  Manual assist for Wordle (single board).

- `quordle.py`  
  Manual assist for Quordle (4 boards, sums entropy across unsolved boards).

- `octordle.py`  
  Manual assist for Octordle (8 boards, sums entropy across unsolved boards).

---

## Requirements

- Python 3.8+ (recommended)
- No external packages needed (standard library only)

---

## Note

1. Change the word list in `dictionary.txt` to input a different set of words
2. Delete `first_guess_entropies.txt` if you want to force a rebuild.\

---

## How patterns work

The programs use Wordle-style feedback encoded as digits:

- `2` = green (correct letter, correct spot)
- `1` = yellow (correct letter, wrong spot)
- `0` = gray (letter not present)

Example: `raise 01200`
