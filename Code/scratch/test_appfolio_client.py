import unittest
from unittest.mock import patch, Mock
import requests
from appfolio_client import AppFolioClient, AppFolioError, to_dataframe


def resp(status=200, body=None, headers=None):
    r = Mock(status_code=status, headers=headers or {}, text='')
    r.json.return_value = body
    return r


def client(**kw):
    session = Mock(headers={})
    args = dict(client_id='ID', client_secret='FAKE_SECRET', developer_id='DEV',
                database='demo', session=session)
    return AppFolioClient(**{**args, **kw}), session


class AppFolioTests(unittest.TestCase):
    def test_database_api_paginates_and_sends_headers(self):
        c, s = client()
        s.request.side_effect = [
            resp(body={'data': [{'Id': 1}], 'next_page_path': '/api/v0/properties?page[number]=2'}),
            resp(body={'data': [{'Id': 2}]}),
        ]
        rows = c.database_api('properties', {'page[size]': 1})
        self.assertEqual(rows, [{'Id': 1}, {'Id': 2}])
        self.assertEqual(s.auth, ('ID', 'FAKE_SECRET'))
        first, second = s.request.call_args_list
        self.assertEqual(first.args, ('GET', 'https://api.appfolio.com/api/v0/properties'))
        self.assertEqual(first.kwargs['headers']['X-AppFolio-Developer-ID'], 'DEV')
        self.assertEqual(second.args[1], 'https://api.appfolio.com/api/v0/properties?page[number]=2')
        self.assertIsNone(second.kwargs['params'])

    def test_report_follows_next_page_url_without_body(self):
        c, s = client()
        s.request.side_effect = [
            resp(body={'results': [{'a': 1}], 'next_page_url': 'https://demo.appfolio.com/next'}),
            resp(body={'results': [{'a': 2}], 'next_page_url': None}),
        ]
        rows = c.report('rent_roll', {'property_visibility': 'all'})
        self.assertEqual(rows, [{'a': 1}, {'a': 2}])
        first, second = s.request.call_args_list
        self.assertEqual(first.args, ('POST', 'https://demo.appfolio.com/api/v2/reports/rent_roll.json'))
        self.assertEqual(first.kwargs['json'], {'property_visibility': 'all'})
        self.assertNotIn('json', second.kwargs)

    def test_unpaginated_report_returns_bare_list(self):
        c, s = client()
        s.request.return_value = resp(body=[{'a': 1}, {'a': 2}])
        self.assertEqual(len(c.report('rent_roll', {'paginate_results': False})), 2)

    @patch('appfolio_client.time.sleep')
    def test_retries_on_429(self, sleep):
        c, s = client()
        s.request.side_effect = [resp(429, headers={'Retry-After': '3'}), resp(body={'data': []})]
        self.assertEqual(c.database_api('units'), [])
        sleep.assert_called_once_with(3.0)

    def test_errors(self):
        for status in (400, 401, 403, 404):
            c, s = client()
            s.request.return_value = resp(status)
            with self.assertRaisesRegex(AppFolioError, str(status)):
                c.database_api('units')
        c, s = client()
        s.request.side_effect = requests.Timeout()
        with self.assertRaisesRegex(AppFolioError, 'timed out'):
            c.database_api('units')

    def test_missing_config_and_secret_not_leaked(self):
        with patch.dict('os.environ', {}, clear=True):
            c, s = client(developer_id=None, database=None)
        with self.assertRaisesRegex(AppFolioError, 'DEVELOPER_ID'):
            c.database_api('units')
        with self.assertRaisesRegex(AppFolioError, 'DATABASE'):
            c.report('rent_roll')
        s.request.assert_not_called()
        self.assertNotIn('FAKE_SECRET', repr(c))

    def test_dataframe_flattens_nested(self):
        df = to_dataframe([{'Id': 1, 'Address': {'City': 'SF'}}])
        self.assertEqual(df.loc[0, 'Address.City'], 'SF')
        self.assertTrue(to_dataframe([]).empty)


if __name__ == '__main__':
    unittest.main()
