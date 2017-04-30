/*
 * James Brundege
 * Date: 2017-04-25
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;

/**
 *
 */
public class FullTrainingSetBatchingStrategy implements BatchingStrategy {

    @Override
    public BatchedDataSet getBatchedDataSet(DataSet trainingSet) {
        return new BatchedDataSet(trainingSet, trainingSet.numRows());
    }
}
