/*
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class BatchedDataSet {

    private final List<DataSet> batches = new ArrayList<>();
    private final Iterator<DataSet> batchIterator;

    public BatchedDataSet(DataSet dataSet, int instancesPerBatch) {
        DataSet shuffledDataSet = dataSet.shuffleRows();
        int numBatches = (int)FastMath.round((double)shuffledDataSet.numRows() / (double) instancesPerBatch);
        int numPerBatch = (int)FastMath.ceil(shuffledDataSet.numRows() / numBatches);
        for (int i = 0; i < numBatches; i++) {
            int start = i*numPerBatch;
            int end = start+numPerBatch;
            if (end > shuffledDataSet.numRows()) {
                end = shuffledDataSet.numRows();
            }
            batches.add(shuffledDataSet.sliceRows(start, end));
        }
        batchIterator = batches.iterator();
    }

    public DataSet getNextBatch() {
        return batchIterator.next();
    }

    public boolean hasNext() {
        return batchIterator.hasNext();
    }

    public int size() {
        return batches.size();
    }
}
