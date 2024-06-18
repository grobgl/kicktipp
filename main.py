import itertools
from typing import Iterator

import numpy as np

test_input = """
Croatia
Draw
Albania
1-0

6.25
2-0

6.5
2-1

9
3-0

10
3-1

15
4-0

21
4-1

26
3-2

34
5-0

56
5-1

67
4-2

67
6-0

126
5-2

151
6-1

201
6-3

201
7-3

201
7-2

201
10-0

201
6-4

201
9-0

201
8-2

201
8-1

201
9-1

201
4-3

251
6-2

251
7-0

251
7-1

301
5-3

301
8-0

301
5-4

501
1-1

8.5
0-0

12.5
2-2

23
3-3

126
5-5

251
4-4

251
1-0

17
2-1

26
2-0

41
3-2

81
3-1

81
3-0

151
8-1

201
8-2

201
6-4

201
7-3

201
8-0

201
6-3

201
7-2

201
4-1

251
4-2

276
4-3

301
4-0

351
5-1

501
5-2

501
5-0

751
5-3

751
6-0

1001
5-4

1001
6-1

1001
7-0

1001
7-1

1001
6-2

1001
"""


def _get_odds(odds: list[str]) -> Iterator[tuple[[tuple[int, int]], float]]:
    for score, _, factor in itertools.batched(odds, 3):
        home, away = score.split("-")
        yield (int(home), int(away)), float(factor)


def parse_odds(odds_str: str) -> dict[tuple[int, int], float]:
    home_team, _, away_team, *odds = odds_str.strip().splitlines()
    grid = np.empty((12, 12))
    grid[:] = np.nan
    scores = _get_odds(odds)
    try:
        score, factor = next(scores)
        while score[0] > score[1]:
            grid[score] = factor
            score, factor = next(scores)
        while True:
            grid[score[::-1]] = factor
            score, factor = next(scores)
    except StopIteration:
        return grid


def get_probabilities(odds: np.array) -> np.array:
    probs = 1 / odds
    probs = probs / np.nansum(probs)
    return np.nan_to_num(probs)


def _expected_points(probs: np.array, score: tuple[int, int]) -> float:
    home, away = score
    if home != away:
        return (
            probs[score] +  # correct score
            np.sum(np.diag(probs, away - home)) +  # same difference
            2 * np.sum(np.tril(probs, -1) if home > away else np.triu(probs, 1))  # same winner
        )
    else:
        return (
            2 * probs[score] +  # correct score
            2 * np.sum(np.diag(probs), away - home)  # same difference
        )


def expected_points(probs: np.array) -> np.array:
    res = np.zeros((12, 12))
    for home in range(12):
        for away in range(12):
            res[(home, away)] = _expected_points(probs, (home, away))
    return res


def max_exp_points(instr: str):
    odds = parse_odds(instr)
    probs = get_probabilities(odds)
    exp = expected_points(probs)
    return np.unravel_index(np.argmax(exp), exp.shape)
