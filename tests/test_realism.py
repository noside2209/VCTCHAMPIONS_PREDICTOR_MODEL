"""Guard rails: simulated maps and box scores must stay inside what pro VALORANT produces.

Reference ranges are approximate VCT (2024-2026) figures: ~20-30% of maps end with the loser on
5 or fewer rounds, ~10% go to overtime, ~21 rounds per map, and per-map player lines rarely leave
rating 0.2-2.6, ACS ~60-450, K/D up to ~7 on a stomp.
"""
import random
import statistics as st
import unittest
from itertools import combinations

from vct_predictor.data import load_dataset
from vct_predictor.players import PlayerModel
from vct_predictor.ratings import RatingModel
from vct_predictor.simulate import simulate_series


class RealismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = load_dataset()
        model, pm = RatingModel(ds), PlayerModel(ds)
        rng = random.Random(2026)
        pairs = list(combinations(ds.teams, 2))
        cls.maps, cls.lines = [], []
        for i in range(1200):
            a, b = pairs[i % len(pairs)]
            s = simulate_series(model, a, b, 3, rng=rng, players=pm)
            for m in s.maps:
                cls.maps.append(m.score)
                cls.lines += m.lines_a + m.lines_b

    def share(self, pred):
        return sum(1 for x in self.maps if pred(x)) / len(self.maps)

    def test_map_score_shape(self):
        self.assertTrue(0.17 <= self.share(lambda x: min(x) <= 5) <= 0.32, "blowout rate")
        self.assertTrue(0.07 <= self.share(lambda x: max(x) > 13) <= 0.15, "overtime rate")
        self.assertTrue(self.share(lambda x: max(x) == 13 and min(x) <= 2) <= 0.05, "13-2 or worse")
        self.assertTrue(20 <= st.mean(sum(x) for x in self.maps) <= 22.5, "rounds per map")

    def test_player_lines(self):
        ratings = [ln.rating for ln in self.lines]
        self.assertTrue(0.2 <= min(ratings) and max(ratings) <= 2.6)
        q = st.quantiles(ratings, n=20)
        self.assertTrue(0.45 <= q[0] <= 0.65 and 1.35 <= q[-1] <= 1.7, "rating 5th/95th percentile")
        self.assertLessEqual(max(ln.acs for ln in self.lines), 470)
        self.assertLessEqual(max(ln.kills / max(ln.deaths, 1) for ln in self.lines), 7.0)
        for ln in self.lines:
            self.assertLessEqual(ln.deaths, ln.rounds)
            self.assertLessEqual(ln.assists, 0.75 * ln.rounds + 1)
            self.assertLessEqual(ln.fk, 0.5 * ln.rounds + 1)


if __name__ == "__main__":
    unittest.main()
