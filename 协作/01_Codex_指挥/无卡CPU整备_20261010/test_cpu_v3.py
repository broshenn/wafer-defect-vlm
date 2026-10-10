import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from core_v3 import load_geometry, measure_v3, hour_sector, reference, parse, schema_errors, score, WaferV3Reward


class SemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = load_geometry()

    def ref(self, mode='eccentric', cls='Loc'):
        return reference(measure_v3(self.g._disc(red_mode=mode)), cls)

    def test_frozen_geometry_regressions(self):
        report = self.g.self_test()
        self.assertTrue(report['all_passed'], report)

    def test_coverage_boundaries(self):
        mapping = [(0, 0, 'none'), (.049999, 1, 'low'), (.05, 1, 'medium'),
                   (.2, 1, 'high'), (.5, 1, 'near_all'), (1, 1, 'near_all')]
        for fraction, n, expected in mapping:
            self.assertEqual(self.g.coverage_level_of(fraction, n)[0], expected)
        obj = self.ref()
        g = measure_v3(self.g._disc())
        g['coverage_level'] = 'near_all'
        self.assertEqual(reference(g, 'Center')['coverage_band'], 'half_or_more')
        self.assertNotIn('caption_zh', obj)

    def test_none_and_red_are_independent(self):
        g = measure_v3(self.g._disc(red_mode='scatter'))
        self.assertGreater(g['n_red'], 0)
        self.assertNotEqual(reference(g, 'none')['coverage_band'], 'zero')

    def test_zero_red(self):
        m = self.g._disc()
        m[m == 2] = 1
        g = measure_v3(m)
        self.assertEqual(reference(g, 'none')['coverage_band'], 'zero')
        self.assertEqual(reference(g, 'none')['red_mass_zones'], [])

    def test_disconnected_opposed_is_not_line(self):
        m = self.g._disc()
        m[m == 2] = 1
        ii, jj = np.indices(m.shape)
        c = (m.shape[0] - 1) / 2
        red = ((ii - c)**2 + (jj - (c + 12))**2 < 9) | ((ii - c)**2 + (jj - (c - 12))**2 < 9)
        m[red] = 2
        self.assertEqual(m[int(c), int(c)], 1)
        ref = reference(measure_v3(m), 'Loc')
        self.assertEqual(ref['angular_type'], 'opposed')
        self.assertNotIn('morphology', ref)

    def test_rotations_clockwise(self):
        m = self.g._disc(red_mode='eccentric')
        r = reference(measure_v3(m), 'Loc')
        self.assertEqual(r['clock_sectors'], [3])
        for k in range(1, 4):
            rotated = reference(measure_v3(np.rot90(m, -k)), 'Loc')
            expected = [(s - 1 + 3*k) % 12 + 1 for s in r['clock_sectors']]
            self.assertEqual(rotated['clock_sectors'], expected)
            self.assertEqual(rotated['coverage_band'], r['coverage_band'])
            self.assertEqual(rotated['red_mass_zones'], r['red_mass_zones'])

    def test_cardinal_clock_and_boundary(self):
        self.assertEqual([hour_sector(x) for x in [0, 90, 180, 270]], [12, 3, 6, 9])
        self.assertEqual(hour_sector(359), 12)
        self.assertEqual(hour_sector(16), 1)

    def test_perfect_reward(self):
        ref = self.ref()
        self.assertAlmostEqual(score(json.dumps(ref), ref)['reward'], 1)

    def test_unknown_mask_is_not_prediction_controlled(self):
        ref = self.ref()
        pred = dict(ref, angular_type='unknown', clock_sectors=[])
        r = score(json.dumps(pred), ref)
        self.assertGreater(r['eligible_weights']['angular'], 0)
        self.assertLess(r['reward'], 1)
        unknown = dict(ref, angular_type='unknown', clock_sectors=[])
        self.assertEqual(score(json.dumps(unknown), unknown)['eligible_weights']['angular'], 0)

    def test_malformed_outputs(self):
        ref = self.ref()
        bad = [json.dumps(ref) + '{}', json.dumps(ref) + ' trailing', '```json\n'+json.dumps(ref)+'\n```',
               '{"defect_class":"Loc","defect_class":"Scratch"}', '{"a":NaN}', '[]']
        for text in bad:
            self.assertEqual(score(text, ref)['reward'], 0)

    def test_bool_duplicate_and_sector_consistency(self):
        obj = self.ref()
        for sectors in [[True], [3, 3], [0], [13], [3.0]]:
            self.assertTrue(schema_errors(dict(obj, clock_sectors=sectors)))
        self.assertTrue(schema_errors(dict(obj, angular_type='opposed', clock_sectors=[1, 3])))
        self.assertTrue(schema_errors(dict(obj, red_mass_zones=['edge', 'edge'])))

    def test_reference_contract_fails_closed(self):
        rw = WaferV3Reward()
        with self.assertRaises(RuntimeError):
            rw(['{}'])
        with self.assertRaises(RuntimeError):
            rw(['{}'], [], ['x'])
        with self.assertRaises(ValueError):
            score('{}', dict(self.ref(), defect_class='unknown'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
