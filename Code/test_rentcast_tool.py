import unittest
from unittest.mock import patch, Mock
import requests
from rentcast_tool import build_params, get_rent_comparables, RentCastError, _parse_response

ARGS = dict(address='123 Example St, Provo, UT 84601', unit='2A', bedrooms=0,
            bathrooms=1, square_feet=600, api_key='FAKE_TEST_KEY')

class RentalTests(unittest.TestCase):
    def test_unit_and_studio(self):
        args = {k: v for k, v in ARGS.items() if k != 'api_key'}
        params = build_params(**args)
        self.assertEqual(params['address'], '123 Example St, Unit 2A, Provo, UT 84601')
        self.assertEqual(params['bedrooms'], 0)
        self.assertEqual(params['lookupSubjectAttributes'], 'false')

    @patch('rentcast_tool.requests.get')
    def test_request_and_numeric_summary(self, get):
        get.return_value = Mock(status_code=200)
        get.return_value.json.return_value = {'rent': 1300, 'comparables': [
            {'price': 1400, 'squareFootage': 700, 'correlation': .8, 'status': 'Inactive'},
            {'price': 1200, 'squareFootage': 600, 'correlation': .9, 'status': 'Active'},
            {'price': None, 'squareFootage': 0}]}
        result = get_rent_comparables(**ARGS)
        self.assertEqual(result['summary']['median_comp_asking_rent'], 1300)
        self.assertEqual(result['summary']['median_active_asking_rent'], 1200)
        self.assertEqual(result['comparables'].iloc[0]['rent_per_sqft'], 2)
        self.assertEqual(get.call_args.kwargs['headers']['X-Api-Key'], 'FAKE_TEST_KEY')
        self.assertNotIn('FAKE_TEST_KEY', str(result))
        get.assert_called_once()

    @patch('rentcast_tool.requests.get')
    def test_missing_unit_attributes_no_network(self, get):
        for value in (None, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                get_rent_comparables(**{**ARGS, 'square_feet': value})
        get.assert_not_called()

    @patch('rentcast_tool.requests.get')
    def test_failures(self, get):
        for status in (400, 401, 403, 404, 429, 500):
            get.return_value = Mock(status_code=status)
            with self.assertRaisesRegex(RentCastError, str(status)):
                get_rent_comparables(**ARGS)
        get.side_effect = requests.Timeout()
        with self.assertRaisesRegex(RentCastError, 'timed out'):
            get_rent_comparables(**ARGS)

    @patch('rentcast_tool.requests.get')
    def test_bad_json(self, get):
        get.return_value = Mock(status_code=200)
        get.return_value.json.side_effect = ValueError()
        with self.assertRaisesRegex(RentCastError, 'invalid JSON'):
            get_rent_comparables(**ARGS)

    def test_empty_and_malformed_results(self):
        params = build_params('1 Main St, Provo, UT 84601', bedrooms=1, bathrooms=1, square_feet=700)
        result = _parse_response({'comparables': []}, params)
        self.assertIsNone(result['summary']['median_comp_asking_rent'])
        self.assertTrue(result['comparables'].empty)
        for payload in ({}, {'comparables': [None]}, []):
            with self.assertRaises(RentCastError):
                _parse_response(payload, params)

if __name__ == '__main__':
    unittest.main()
