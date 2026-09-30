import unittest
from itertools import permutations

from vct_predictor.data import load_dataset
from vct_predictor.ratings import RatingModel
from vct_predictor.scores import scoreline_distribution
from vct_predictor.series import predict_series, series_score_distribution
from vct_predictor.tournament import PLACEMENTS, Tournament
from vct_predictor.veto import veto_sequence


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset()
        cls.model = RatingModel(cls.ds)

    def test_sixteen_teams_four_groups(self):
        self.assertEqual(len(self.ds.teams), 16)
        grouped = [t for g in self.ds.event["groups"].values() for t in g]
        self.assertEqual(sorted(grouped), sorted(self.ds.teams))

    def test_scoreline_distribution_matches_target(self):
        for p in (0.3, 0.5, 0.72):
            d = scoreline_distribution(p)
            self.assertAlmostEqual(sum(d.values()), 1.0, places=6)
            self.assertAlmostEqual(sum(v for (a, b), v in d.items() if a > b), p, places=2)
            for a, b in d:
                self.assertTrue((max(a, b) == 13 and min(a, b) <= 11) or abs(a - b) == 2)

    def test_series_distribution(self):
        d = series_score_distribution([0.5, 0.5, 0.5], 3)
        self.assertAlmostEqual(d[(2, 0)], 0.25)
        self.assertAlmostEqual(d[(2, 1)], 0.25)
        self.assertAlmostEqual(sum(series_score_distribution([0.6] * 5, 5).values()), 1.0)

    def test_veto_formats(self):
        self.assertEqual(veto_sequence(3, 7), ["A_ban", "B_ban", "A_pick", "B_pick", "A_ban", "B_ban", "decider"])
        self.assertEqual(veto_sequence(5, 7), ["A_ban", "B_ban", "A_pick", "B_pick", "A_pick", "B_pick", "decider"])
        self.assertEqual(len(veto_sequence(1, 7)), 7)

    def test_every_matchup_is_valid_and_symmetric(self):
        names = list(self.ds.teams)
        for a, b in permutations(names, 2):
            for bo in (1, 3, 5):
                p = predict_series(self.model, a, b, bo)
                self.assertEqual(len(p.maps), bo)
                self.assertEqual(len({m.map for m in p.maps}), bo)
                self.assertAlmostEqual(sum(p.score_dist.values()), 1.0, places=9)
                self.assertTrue(set(m.map for m in p.maps) <= set(self.model.pool))
        for m in self.model.pool:
            pa = self.model.map_win_prob("Paper Rex", "NRG", m)
            pb = self.model.map_win_prob("NRG", "Paper Rex", m)
            self.assertAlmostEqual(pa + pb, 1.0, places=9)

    def test_custom_pool_and_home(self):
        pool = ["Ascent", "Haven", "Lotus", "Split", "Sunset"]
        p = predict_series(self.model, "TYLOO", "T1", 3, pool, home_region="China")
        self.assertTrue(all(m.map in pool for m in p.maps))
        neutral = predict_series(self.model, "TYLOO", "T1", 3, pool)
        self.assertGreater(p.p_a, neutral.p_a)

    def test_tournament_respects_locked_results(self):
        tour = Tournament(self.ds, self.model)
        log, placement = tour.chalk()
        by_id = {r.match_id: r for r in log}
        self.assertEqual(by_id["C-winners"].winner, "Paper Rex")
        self.assertEqual(by_id["D-winners"].winner, "NRG")
        self.assertEqual(sorted(placement), sorted(self.ds.teams))
        sims = tour.simulate(300, seed=1)
        self.assertAlmostEqual(sum(v["win"] for v in sims["teams"].values()), 1.0)
        self.assertEqual(sims["teams"]["Paper Rex"]["playoffs"], 1.0)
        for v in sims["teams"].values():
            self.assertAlmostEqual(sum(v["placements"][p] for p in PLACEMENTS), 1.0)


if __name__ == "__main__":
    unittest.main()
