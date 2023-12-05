from __future__ import division
import numpy as np
import random,pdb
import operator

def roulette_selection(weights):
        '''performs weighted selection or roulette wheel selection on a list
        and returns the index selected from the list'''

        # sort the weights in ascending order
        sorted_indexed_weights = sorted(enumerate(weights), key=operator.itemgetter(1));
        indices, sorted_weights = zip(*sorted_indexed_weights);
        # calculate the cumulative probability
        tot_sum=sum(sorted_weights)
        prob = [x/tot_sum for x in sorted_weights]
        cum_prob=np.cumsum(prob)
        # select a random a number in the range [0,1]
        random_num=random.random()

        for index_value, cum_prob_value in zip(indices,cum_prob):
            if random_num < cum_prob_value:
                return index_value


if __name__ == "__main__":
    # weights=[1,2,6,4,3,7,20]
    weights = [0.1, 0.3, 0.4, 0.2]
    print (roulette_selection(weights))
    # weights=[1,2,2,2,2,2,2]
    weights = [0.9, 0.05, 0.05]
    print (roulette_selection(weights))