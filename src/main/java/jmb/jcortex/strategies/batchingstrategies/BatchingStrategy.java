/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;

public interface BatchingStrategy {

    BatchedDataSet getBatchedDataSet(DataSet trainingSet);

}
