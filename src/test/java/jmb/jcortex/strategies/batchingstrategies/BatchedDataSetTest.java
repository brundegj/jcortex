/*
 * James Brundege
 * Date: 2017-04-17
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import org.junit.Test;

import static java.lang.String.format;
import static org.assertj.core.api.Assertions.assertThat;

public class BatchedDataSetTest {

    @Test
    public void batchedDataSet_RoundsTheSplitToMinizeSizeDifferences() {
        BatchedDataSet batchedDataSet = new BatchedDataSet(createDataSet(24), 8);
        assertThat(batchedDataSet.size()).isEqualTo(3);
        assertBatchesHaveSize(batchedDataSet, 8);

        batchedDataSet = new BatchedDataSet(createDataSet(25), 8);
        assertThat(batchedDataSet.size()).isEqualTo(3);
        assertBatchesHaveSize(batchedDataSet, 8, 9);

        batchedDataSet = new BatchedDataSet(createDataSet(26), 8);
        assertThat(batchedDataSet.size()).isEqualTo(3);
        assertBatchesHaveSize(batchedDataSet, 8, 9);

        batchedDataSet = new BatchedDataSet(createDataSet(27), 8);
        assertThat(batchedDataSet.size()).isEqualTo(3);
        assertBatchesHaveSize(batchedDataSet, 8, 9);

        batchedDataSet = new BatchedDataSet(createDataSet(28), 8);
        assertThat(batchedDataSet.size()).isEqualTo(4);
        assertBatchesHaveSize(batchedDataSet, 7, 8);

        batchedDataSet = new BatchedDataSet(createDataSet(29), 8);
        assertThat(batchedDataSet.size()).isEqualTo(4);
        assertBatchesHaveSize(batchedDataSet, 7, 8);

        batchedDataSet = new BatchedDataSet(createDataSet(30), 8);
        assertThat(batchedDataSet.size()).isEqualTo(4);
        assertBatchesHaveSize(batchedDataSet, 7, 8);

        batchedDataSet = new BatchedDataSet(createDataSet(31), 8);
        assertThat(batchedDataSet.size()).isEqualTo(4);
        assertBatchesHaveSize(batchedDataSet, 7, 8);

        batchedDataSet = new BatchedDataSet(createDataSet(32), 8);
        assertThat(batchedDataSet.size()).isEqualTo(4);
        assertBatchesHaveSize(batchedDataSet, 8);
    }

    private DataSet createDataSet(int numRows) {
        double[][] values = new double[numRows][1];
        for (int i = 0; i < values.length; i++) {
            values[i] = new double[]{i};
        }
        return new DataSet(new SynMatrix(values), new SynMatrix(values));
    }

    private void assertBatchesHaveSize(BatchedDataSet batchedDataSet, int size) {
        assertBatchesHaveSize(batchedDataSet, size, size);
    }

    private void assertBatchesHaveSize(BatchedDataSet batchedDataSet, int option1, int option2) {
        while (batchedDataSet.hasNext()) {
            DataSet batch = batchedDataSet.getNextBatch();
            if (batch.numRows() != option1 && batch.numRows() != option2) {
                throw new AssertionError(format("Expected batches of size %s or %s, but got a batch of size %s",
                        option1, option2, batch.numRows()));
            }
        }
    }

}