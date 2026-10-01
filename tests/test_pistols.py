import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from vct_predictor.data import load_dataset
from vct_predictor.ratings import RatingModel
from vct_predictor.scores import NEUTRAL, play_map, scoreline_distribution
from vct_predictor.series import predict_series
from vct_predictor.simulate import simulate_series
from vct_predictor.vlr import parse_match_rounds, tally_sides

ROOT = Path(__file__).resolve().parent.parent


class PistolTests(unittest.TestCase):
    def test_neutral_model_unchanged_by_pistol_layer(self):
        ds = load_dataset()
        m = RatingModel(ds)
        self.assertFalse(m.sides.has_any_data)
        for a, b in [("LOUD", "Team Vitality"), ("100 Thieves", "Xi Lai Gaming")]:
            for mp in m.pool:
                self.assertEqual(m.map_context(a, b, mp), NEUTRAL)
                self.assertAlmostEqual(m.map_win_prob(a, b, mp), m.elo_map_prob(a, b, mp))
        for p in (0.35, 0.5, 0.7):
            d = scoreline_distribution(p)
            self.assertAlmostEqual(sum(v for (x, y), v in d.items() if x > y), p, places=2)

    def test_simulated_maps_have_two_pistols_and_valid_halves(self):
        rng = random.Random(4)
        for _ in range(300):
            play = play_map(0.6, rng)
            self.assertEqual([n for n, _, _ in play.pistols], [1, 13])
            self.assertNotEqual(play.pistols[0][2], play.pistols[1][2])  # sides swap at half
            self.assertEqual(sum(play.half), 12)

    def test_vlr_match_parser_and_tally(self):
        games = parse_match_rounds((ROOT / "tests" / "fixtures" / "vlr_match.html").read_text())
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["map"], "Ascent")
        rec = tally_sides(games, {"LOUD": "LOUD", "Team Vitality": "VIT"}, ["Ascent"])["teams"]
        self.assertEqual(rec["LOUD"]["Ascent"]["def_pistol"], [1, 1])
        self.assertEqual(rec["LOUD"]["Ascent"]["atk_pistol"], [0, 1])
        self.assertEqual(rec["Team Vitality"]["Ascent"]["def_pistol"], [1, 1])
        self.assertEqual(rec["LOUD"]["Ascent"]["def_rounds"], [2, 3])

    def test_pistol_data_moves_predictions(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for f in (ROOT / "data").glob("*.json"):
                shutil.copy(f, tmp / f.name)
            side = json.loads((tmp / "side_stats.json").read_text())
            side["teams"] = {"Team Vitality": {"Haven": {"atk_pistol": [18, 20], "def_pistol": [17, 20],
                                                          "atk_rounds": [120, 220], "def_rounds": [110, 220]}},
                             "LOUD": {"Haven": {"atk_pistol": [4, 20], "def_pistol": [5, 20],
                                                "atk_rounds": [100, 220], "def_rounds": [115, 220]}}}
            (tmp / "side_stats.json").write_text(json.dumps(side))
            m0 = RatingModel(load_dataset())
            m1 = RatingModel(load_dataset(tmp))
            base = m0.map_win_prob("Team Vitality", "LOUD", "Haven")
            boosted = m1.map_win_prob("Team Vitality", "LOUD", "Haven")
            self.assertGreater(boosted, base + 0.03)  # strong pistol record -> real but bounded edge
            self.assertLess(boosted, base + 0.15)
            ctx = m1.map_context("Team Vitality", "LOUD", "Haven")
            self.assertGreater(ctx.pistol_atk, 0.6)
            self.assertAlmostEqual(m1.map_context("LOUD", "Team Vitality", "Haven").pistol_def, 1 - ctx.pistol_atk, places=6)
            pred = predict_series(m1, "Team Vitality", "LOUD", 3)
            for mp in pred.maps:
                if mp.map == "Haven":
                    self.assertGreater(mp.p_pistol_atk, 0.6)
            s = simulate_series(m1, "Team Vitality", "LOUD", 3, rng=random.Random(1))
            self.assertTrue(all(len(x.pistols) == 2 for x in s.maps))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
