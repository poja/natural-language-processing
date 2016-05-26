import unittest

import lexical_disambiguation


class DisambiguationTest(unittest.TestCase):

    def test_text_bayes_probabilities(self):
        classifier = lexical_disambiguation.TextBayes(smoothing='add-one')
        classifier.train(
            ['food', 'food yummy', 'yummy food', 'bad', 'bad thief'],
            ['cooking', 'cooking', 'cooking', 'fiction', 'fiction'])
        probabilities = classifier.belong_probabilities('nice yummy food')

        cooking_food_probability = (3 + 1) / (5 + 2)
        cooking_yummy_probability = (2 + 1) / (5 + 2)
        fiction_bad_probability = (2 + 1) / (3 + 2)
        fiction_thief_probability = (1 + 1) / (3 + 2)

        cooking_unknown_probability = 1 / (5 + 2)
        fiction_unknown_probability = 1 / (3 + 2)

        cooking_prior = 3 / 5
        fiction_prior = 2 / 5

        cooking_likelihood = cooking_unknown_probability * cooking_yummy_probability * cooking_food_probability * \
                             cooking_prior
        fiction_likelihood = fiction_unknown_probability ** 3 * fiction_prior

        final_cooking_probability = cooking_likelihood / (cooking_likelihood + fiction_likelihood)
        final_fiction_probability = fiction_likelihood / (cooking_likelihood + fiction_likelihood)

        self.assertAlmostEqual(probabilities['cooking'], final_cooking_probability)
        self.assertAlmostEqual(probabilities['fiction'], final_fiction_probability)
