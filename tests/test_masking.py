import random
import unittest

from ru_pii.schema import Entity
from src.core.masking import MaskedSpan, mask_text, unmask_text
from src.core.postprocess import filter_public
from src.models.hybrid import HybridDetector


class MaskingTests(unittest.TestCase):
    def test_unicode_overlap_roundtrip(self):
        text = '🙂 Иван\nпочта: a@b.ru; e\u0301 *'
        spans = [Entity(2, 6, 'PERSON', 1, ''), Entity(3, 5, 'PERSON', 1, ''),
                 Entity(14, 20, 'EMAIL', 1, ''), Entity(18, 22, 'OTHER_ID', 1, '')]
        masked, saved = mask_text(text, spans)
        self.assertEqual(len(masked), len(text))
        self.assertEqual(masked[:2], text[:2])
        self.assertEqual(unmask_text(masked, saved), text)

    def test_randomized_union_and_roundtrip(self):
        rng = random.Random(42)
        for _ in range(250):
            text = ''.join(rng.choices('Иван🙂* \n\u0301', k=100))
            entities = []
            expected = list(text)
            for _ in range(30):
                start = rng.randrange(len(text))
                end = rng.randrange(start + 1, len(text) + 1)
                entities.append(Entity(start, end, 'PERSON', 1, ''))
                expected[start:end] = '*' * (end - start)
            masked, spans = mask_text(text, entities)
            self.assertEqual(masked, ''.join(expected))
            self.assertEqual(unmask_text(masked, spans), text)

    def test_invalid_offsets_fail_closed(self):
        for start, end in [(-1, 2), (0, 5), (2, 2), (True, 2), (0.5, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                mask_text('abc', [Entity(start, end, 'PERSON', 1, '')])

    def test_corrupt_restoration_rejected(self):
        for spans in [[MaskedSpan(0, 2, 'abc', 'PERSON')],
                      [MaskedSpan(0, 2, 'ab', 'PERSON'), MaskedSpan(1, 3, 'xy', 'PERSON')]]:
            with self.assertRaises(ValueError):
                unmask_text('***', spans)

    def test_auxiliary_public_label_does_not_mask_or_whitelist_personal(self):
        public = Entity(0, 4, 'PUBLIC_PERSON', 1, 'Иван')
        private = Entity(0, 4, 'PERSON', 1, 'Иван')
        self.assertEqual(mask_text('Иван', [public])[0], 'Иван')
        self.assertEqual(mask_text('Иван', [public, private])[0], '****')

    def test_occupation_is_not_public_consent(self):
        text = 'Врач Иванов живёт рядом с банком'
        person = Entity(5, 11, 'PERSON', 1, 'Иванов')
        self.assertEqual(filter_public(text, [person]), [person])

    def test_hybrid_retains_model_pattern_missed_by_rules(self):
        entity = Entity(0, 3, 'EMAIL', 1, 'a@b')
        class Model:
            def predict(self, text):
                return [entity]
        self.assertIn(entity, HybridDetector(Model()).predict('a@b'))

    def test_partial_email_prediction_is_completed(self):
        from src.models.rules import supplement_emails
        text = 'Контакт: test@example.com'
        partial = Entity(13, len(text), 'EMAIL', 0.9, text[13:])
        entities = supplement_emails(text, [partial])
        masked, spans = mask_text(text, entities)
        self.assertEqual(masked, 'Контакт: ' + '*' * len('test@example.com'))
        self.assertEqual(unmask_text(masked, spans), text)

    def test_email_supplement_keeps_other_entities(self):
        from src.models.rules import supplement_emails
        text = 'Иван test@example.com'
        entities = [Entity(0, 4, 'PERSON', 1, 'Иван')]
        output = supplement_emails(text, entities)
        self.assertIn(entities[0], output)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(supplement_emails(text, output)), 2)
