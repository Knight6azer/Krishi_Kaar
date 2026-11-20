import unittest
import translations

class TestTranslations(unittest.TestCase):
    def test_languages_present(self):
        expected_langs = {'en', 'hi', 'mr', 'gu', 'ta', 'te', 'kn', 'ur'}
        self.assertTrue(expected_langs.issubset(translations.translations.keys()))

    def test_translation_key_parity(self):
        en_keys = set(translations.translations['en'].keys())
        for lang, trans_dict in translations.translations.items():
            missing = en_keys - set(trans_dict.keys())
            self.assertEqual(len(missing), 0, f"Language '{lang}' missing keys: {missing}")

    def test_no_empty_translations(self):
        for lang, trans_dict in translations.translations.items():
            for key, val in trans_dict.items():
                self.assertTrue(bool(str(val).strip()), f"Empty translation for key '{key}' in lang '{lang}'")

if __name__ == '__main__':
    unittest.main()
