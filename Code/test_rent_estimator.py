import datetime as dt
import math
import os
import tempfile
import unittest

import rent_estimator as re_

HERE = os.path.dirname(os.path.abspath(__file__))
AS_OF = dt.date(2026, 10, 14)


def unit(**kw):
    u = {"address": "1 Test St", "zip": "94110", "beds": 1, "baths": "1", "sqft": 700,
         **re_.DEFAULTS, "unit_features": [], "building_amenities": []}
    u.update(kw)
    return u


LOG_MODEL = {"form": "log", "smearing": 1.0, "interval": {"p10": -0.1, "p90": 0.1},
             "coefficients": {"intercept": 5.0, "bedrooms": 0.1, "log_sqft": 0.5,
                              "zip:94110": 0.05, "unit:Views": 0.05,
                              "parking:Assigned garage space": 0.06, "unit:Furnished": -0.02}}


class EstimatorTests(unittest.TestCase):
    def test_log_model_base_and_adjustments_reconcile(self):
        m = re_.model_from_dict(LOG_MODEL)
        r = re_.estimate(unit(unit_features=["Views", "Furnished"], parking="Assigned garage space"),
                         m, as_of=AS_OF)
        base = math.exp(5.0 + 0.05 + 0.1 + 0.5 * math.log(700))
        self.assertAlmostEqual(r["base"]["amount"], base, places=6)
        total = base * math.exp(0.05 + 0.06 - 0.02)
        self.assertAlmostEqual(r["unrounded"], total, places=6)
        self.assertAlmostEqual(base + sum(a["amount"] for a in r["adjustments"]), total, places=6)
        self.assertEqual(r["recommended"] % 25, 0)
        self.assertLess(r["low"], r["recommended"])
        self.assertGreater(r["high"], r["recommended"])

    def test_linear_model_adds_dollars(self):
        m = re_.model_from_dict({"form": "linear", "interval": {"sd": 300},
                                 "coefficients": {"intercept": 1000, "bedrooms": 500, "sqft": 2,
                                                  "unit:Views": 200}})
        r = re_.estimate(unit(beds=2, sqft=800, unit_features=["Views"]), m, as_of=AS_OF)
        self.assertEqual(r["base"]["amount"], 1000 + 1000 + 1600)
        self.assertEqual(r["recommended"], 3800)

    def test_bad_coefficient_name_rejected(self):
        with self.assertRaises(ValueError):
            re_.model_from_dict({"coefficients": {"intercept": 1, "unit:Viewz": 0.1}})
        with self.assertRaises(ValueError):
            re_.model_from_dict({"coefficients": {"intercept": 1, "garage": 0.1}})

    def test_overrides_win_over_file(self):
        m = re_.load_model(os.path.join(HERE, "rent_coefficients_example.json"),
                           {"unit:Views": 0.5})
        self.assertEqual(m["coefficients"]["unit:Views"], 0.5)

    def test_loads_trained_rent_model_py(self):
        m = re_.load_model(os.path.join(HERE, "rent_model.py"))
        self.assertIn("zip:94110", m["coefficients"])
        r = re_.estimate(unit(zip="99999"), m, as_of=AS_OF)
        self.assertTrue(any("other" in w for w in r["warnings"]))

    def test_comps_adjusted_and_blended(self):
        m = re_.model_from_dict(LOG_MODEL)
        comps = [{"address": f"{i} Comp St", "unit": "", "zip": "94110", "beds": 1, "baths": 1,
                  "sqft": 700, "rent": 4000, "date": AS_OF, "note": "", "kind": "Signed lease"}
                 for i in range(5)]
        r = re_.estimate(unit(), m, comps, comp_weight=1.0, as_of=AS_OF)
        self.assertEqual(r["confidence"], "high")
        self.assertAlmostEqual(r["comp_median_adjusted"], 4000)
        self.assertEqual(r["recommended"], 4000)

    def test_store_and_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "est.jsonl")
            m = re_.model_from_dict(LOG_MODEL)
            u = unit()
            r = re_.estimate(u, m, as_of=AS_OF)
            i = re_.save_estimate(path, u, r)
            re_.record_actual(path, i, r["recommended"] * 1.02)
            a = re_.accuracy(re_.read_store(path))
            self.assertEqual(a["with_actual"], 1)
            self.assertEqual(a["within_5_pct"], 100)


if __name__ == "__main__":
    unittest.main()
