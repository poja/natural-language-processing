import math
import unittest

import collocations


class CollocationsTest(unittest.TestCase):
    EXAMPLE = 'We like to ride bikes. We ride bikes all day. Riding bikes is fun. Yossi also likes to ride bikes.'

    EXAMPLE_TOK_COUNT = 24
    EXAMPLE_COLL_COUNT = 20
    EXAMPLE_TRI_COUNT = 16

    EXAMPLE_TO_PR = 2. / EXAMPLE_TOK_COUNT
    EXAMPLE_RIDE_PR = 3. / EXAMPLE_TOK_COUNT
    EXAMPLE_BIKES_PR = 4. / EXAMPLE_TOK_COUNT

    EXAMPLE_TORIDE_PR = 2. / EXAMPLE_COLL_COUNT
    EXAMPLE_RIDEBIKES_PR = 3. / EXAMPLE_COLL_COUNT

    EXAMPLE_RIDEBIKES_RAWFREQ = 3. / EXAMPLE_TOK_COUNT
    EXAMPLE_RIDEBIKES_PMI = math.log2(EXAMPLE_RIDEBIKES_PR / (EXAMPLE_RIDE_PR * EXAMPLE_BIKES_PR))

    EXAMPLE_TORIDEBIKES_PR = 2. / EXAMPLE_TRI_COUNT
    EXAMPLE_TORIDEBIKES_PMI_A = math.log2(
        EXAMPLE_TORIDEBIKES_PR / (EXAMPLE_TO_PR * EXAMPLE_RIDE_PR * EXAMPLE_BIKES_PR))
    EXAMPLE_TORIDEBIKES_PMI_B = math.log2(
        EXAMPLE_TORIDEBIKES_PR / (EXAMPLE_TORIDE_PR * EXAMPLE_RIDEBIKES_PR))
    EXAMPLE_TORIDEBIKES_PMI_C = math.log2(
        EXAMPLE_TORIDEBIKES_PR / (
            EXAMPLE_TORIDE_PR * EXAMPLE_RIDEBIKES_PR * EXAMPLE_TO_PR * EXAMPLE_RIDE_PR * EXAMPLE_BIKES_PR))

    def test_right_unigrams_count(self):
        unigrams_count = len(collocations.all_unigrams(CollocationsTest.EXAMPLE))
        self.assertEqual(unigrams_count, CollocationsTest.EXAMPLE_TOK_COUNT)

    def test_spaces_not_in_unigrams(self):
        unigrams = collocations.all_unigrams(CollocationsTest.EXAMPLE)
        self.assertFalse(' ' in unigrams)

    def test_unigram_probability(self):
        unigram_pr = collocations.unigram_probabilities(CollocationsTest.EXAMPLE)
        self.assertEqual(unigram_pr['ride'], CollocationsTest.EXAMPLE_RIDE_PR)

    def test_collocation_rawfreq(self):
        rawfreqs = collocations.collocation_raw_frequencies(CollocationsTest.EXAMPLE)
        self.assertEqual(rawfreqs[('ride', 'bikes')], CollocationsTest.EXAMPLE_RIDEBIKES_RAWFREQ)

    def test_collocation_count(self):
        colls = collocations.all_collocations(CollocationsTest.EXAMPLE)
        self.assertEqual(CollocationsTest.EXAMPLE_COLL_COUNT, len(colls))

    def test_collocation_probability(self):
        col_probs = collocations.collocation_probabilities(CollocationsTest.EXAMPLE)
        ride_bikes_pr = col_probs[('ride', 'bikes')]
        self.assertAlmostEqual(ride_bikes_pr, CollocationsTest.EXAMPLE_RIDEBIKES_PR)

    def test_collocation_pmi(self):
        collocation_pmis = collocations.collocation_pmi_values(CollocationsTest.EXAMPLE)
        self.assertAlmostEqual(collocation_pmis[('ride', 'bikes')], CollocationsTest.EXAMPLE_RIDEBIKES_PMI)

    def test_pmi_filtersout_rarewords(self):
        collocation_pmis = collocations.collocation_pmi_values(CollocationsTest.EXAMPLE, wordcount_filter=3)
        self.assertTrue('bikes', '.' in collocation_pmis)
        self.assertTrue('ride', 'bikes' in collocation_pmis)
        self.assertEquals(len(collocation_pmis.keys()), 2)

    def test_trigram_count(self):
        trigrams = collocations.all_trigrams(CollocationsTest.EXAMPLE)
        self.assertEqual(CollocationsTest.EXAMPLE_TRI_COUNT, len(trigrams))

    def test_trigram_pr(self):
        trigrams_pr = collocations.trigram_probabilities(CollocationsTest.EXAMPLE)
        self.assertEquals(trigrams_pr[('to', 'ride', 'bikes')], CollocationsTest.EXAMPLE_TORIDEBIKES_PR)

    def test_trigram_pmi_a(self):
        trigrams_pmi_a = collocations.trigram_pmi_values(CollocationsTest.EXAMPLE, 'a')
        self.assertAlmostEqual(trigrams_pmi_a[('to', 'ride', 'bikes')], CollocationsTest.EXAMPLE_TORIDEBIKES_PMI_A)

    def test_trigram_pmi_b(self):
        trigrams_pmi_b = collocations.trigram_pmi_values(CollocationsTest.EXAMPLE, 'b')
        self.assertAlmostEqual(trigrams_pmi_b[('to', 'ride', 'bikes')], CollocationsTest.EXAMPLE_TORIDEBIKES_PMI_B)

    def test_trigram_pmi_c(self):
        trigrams_pmi_c = collocations.trigram_pmi_values(CollocationsTest.EXAMPLE, 'c')
        self.assertAlmostEqual(trigrams_pmi_c[('to', 'ride', 'bikes')], CollocationsTest.EXAMPLE_TORIDEBIKES_PMI_C)

    def test_trigram_pmi_filtersout_rarewords(self):
        trigram_pmi_a = collocations.trigram_pmi_values(CollocationsTest.EXAMPLE, 'a', wordcount_filter=2)
        self.assertTrue(('We', 'ride', 'bikes') in trigram_pmi_a)
        self.assertTrue(('to', 'ride', 'bikes') in trigram_pmi_a)
        self.assertTrue(('ride', 'bikes', '.') in trigram_pmi_a)
        self.assertEqual(len(trigram_pmi_a), 3)
