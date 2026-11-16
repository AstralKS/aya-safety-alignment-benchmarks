"""Offline tests of the one pinned, user-approved FActScore selection."""
import io
import unittest
import zipfile

import factscore_selection as selection
import prepare_v2_data as preparation
import freeze_factscore_evidence as freezer


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = (preparation.DEFAULT_ROOT / selection.FACTSCORE_PATH).read_bytes()
        with zipfile.ZipFile(io.BytesIO(cls.archive)) as source:
            cls.original = source.read(selection.FACTSCORE_MEMBER).decode('utf-8').splitlines()

    def test_exact_exclusions_preserve_original_indices_order_and_prompts(self):
        selected, details = selection.select_archive(self.archive)
        exclusions = [(2, 'William Post'), (127, 'Carlos Alfonso'), (162, 'Jagan')]
        self.assertEqual([(r['source_index'], r['topic']) for r in details['excluded_subjects']], exclusions)
        self.assertTrue(all(r['reason'] for r in details['excluded_subjects']))
        expected = [(i, topic) for i, topic in enumerate(self.original) if i not in {2, 127, 162}]
        self.assertEqual(selected, expected)
        self.assertEqual(details['original_count'], 183)
        self.assertEqual(details['raw_count'], 183)
        self.assertEqual(details['selected_count'], 180)
        self.assertEqual(details['selected_source_indices'], [i for i, _ in expected])
        self.assertEqual(details['selected_ids'], [f'v2-factscore_bios-{i}' for i, _ in expected])
        rows, prepared_details = preparation._factscore({selection.FACTSCORE_PATH: self.archive})
        self.assertEqual(prepared_details, details)
        self.assertEqual([(r['metadata']['source_index'], r['metadata']['topic']) for r in rows], expected)
        self.assertEqual([r['question'] for r in rows], [f'Question: Tell me a bio of {topic}.' for _, topic in expected])
        self.assertEqual(freezer.subject_selection(preparation.DEFAULT_ROOT), (selected, details))
        self.assertEqual(freezer.topics(preparation.DEFAULT_ROOT), [topic for _, topic in expected])

    def test_bad_inventory_never_silently_drops_or_renumbers(self):
        cases = [self.original[:-1], self.original + ['Extra subject']]
        duplicate = self.original.copy()
        duplicate[10] = duplicate[9]
        cases.append(duplicate)
        blank = self.original.copy()
        blank[10] = '   '
        cases.append(blank)
        missing_exclusion = self.original.copy()
        missing_exclusion[2] = 'Replacement subject'
        cases.append(missing_exclusion)
        reordered = self.original.copy()
        reordered[2], reordered[3] = reordered[3], reordered[2]
        cases.append(reordered)
        for inventory in cases:
            with self.subTest(inventory=inventory[:4]), self.assertRaises(ValueError):
                selection.select_inventory(inventory)

    def test_archive_hash_cannot_be_redeclared(self):
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            selection.select_archive(self.archive + b'changed')


if __name__ == '__main__':
    unittest.main()
