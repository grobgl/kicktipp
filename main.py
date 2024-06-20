import itertools
from functools import cached_property
from typing import Iterator

import numpy as np


class OddsUtils:
    def __init__(self, home: str, away: str, odds: np.array):
        self._home = home
        self._away = away
        self._odds = odds

    @cached_property
    def probabilities(self) -> np.array:
        probs = 1 / self._odds
        probs = probs / np.nansum(probs)
        return np.nan_to_num(probs)

    @cached_property
    def expected_kicktipp_points(self) -> np.array:
        res = np.zeros((12, 12))
        for home, away in itertools.product(range(12), repeat=2):
            res[(home, away)] = self._expected_kicktipp_points((home, away))
        return res

    @cached_property
    def scores_by_expected_kicktipp_points(self) -> list[tuple[int, int], float]:
        flat_indices = np.arange(self.expected_kicktipp_points.size)
        scores = np.unravel_index(flat_indices, self.expected_kicktipp_points.shape)
        values = self.expected_kicktipp_points[scores]
        result = [((scores[0][i], scores[1][i]), values[i]) for i in range(self.expected_kicktipp_points.size)]
        return sorted(result, key=lambda x: x[1], reverse=True)

    def pretty_print(self):
        print(f"{self._home} - {self._away}:")
        for (home, away), expected_points in self.scores_by_expected_kicktipp_points[:10]:
            print(f"  {home} - {away}: {expected_points: .2f}")

    def _expected_kicktipp_points(self, score: tuple[int, int]) -> float:
        home, away = score
        if home != away:
            return (
                self.probabilities[score] +  # correct score
                np.sum(np.diag(self.probabilities, away - home)) +  # same difference
                2 * np.sum(np.tril(self.probabilities, -1) if home > away else np.triu(self.probabilities, 1))  # same winner
            )
        else:
            return (
                2 * self.probabilities[score] +  # correct score
                2 * np.sum(np.diag(self.probabilities), away - home)  # same difference
            )

    @classmethod
    def from_oddschecker(cls, clip: str) -> "OddsUtils":
        home, _, away, *odds_table = clip.strip().splitlines()
        odds = np.empty((12, 12))
        odds[:] = np.nan
        for score, factor in cls._parse_odds_gen(odds_table):
            odds[score] = factor
        return cls(home, away, odds)

    @staticmethod
    def _parse_odds_gen(odds: list[str]) -> Iterator[tuple[[tuple[int, int]], float]]:
        home_odds = True  # flip home/away once done parsing home wins
        for score, _, factor in itertools.batched(odds, 3):
            try:
                home, away = map(int, score.split("-"))
            except:
                continue
            if home == away:
                home_odds = False
            if not home_odds:
                home, away = away, home
            yield (home, away), float(factor)


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
