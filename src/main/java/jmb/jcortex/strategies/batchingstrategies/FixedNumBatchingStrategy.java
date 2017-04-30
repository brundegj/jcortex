/*
 * James Brundege
 * Date: 2017-04-11
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;

public class FixedNumBatchingStrategy implements BatchingStrategy {

    private int batchSize;

    public FixedNumBatchingStrategy(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public BatchedDataSet getBatchedDataSet(DataSet trainingSet) {
        return new BatchedDataSet(trainingSet, batchSize);
    }

}
