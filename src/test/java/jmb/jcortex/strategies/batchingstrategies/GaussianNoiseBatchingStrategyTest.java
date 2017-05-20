/*
 * James Brundege
 * Date: 2017-05-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.mapfunctions.MatrixFunction;
import org.assertj.core.api.DoubleArrayAssert;
import org.assertj.core.data.Offset;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.Arrays;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@RunWith(MockitoJUnitRunner.class)
public class GaussianNoiseBatchingStrategyTest {
    private Offset<Double> precision = Offset.offset(0.01);

    @Test
    public void testGaussianNoise() {
        // mean of 3, std of 1
        double[] twos = SynMatrix.ones(1, 100).applyInPlace(x -> x * 2).getRow(0);
        double[] fours = SynMatrix.ones(1, 100).applyInPlace(x -> x * 4).getRow(0);
        SynMatrix features = new SynMatrix(1000, 100);
        IntStream.range(0, features.numRows()).forEach(rowNum -> {
             if (rowNum % 2 == 0) {
                 features.setRow(rowNum, twos);
             } else {
                 features.setRow(rowNum, fours);
             }
        });

        DataSet trainingSet = new DataSet(features, null);

        GaussianNoiseBatchingStrategy batchingStrategy = new GaussianNoiseBatchingStrategy(500, 1, trainingSet);

        BatchedDataSet batchedDataSet = batchingStrategy.getBatchedDataSet(trainingSet);
        DataSet first = batchedDataSet.getNextBatch();
        Arrays.stream(first.getFeatures().getAll()).forEach(x -> {
            assertThat(x).isNotEqualTo(3.0);
        });

        double[] actualMeans = first.getFeatures().getColMeans().getCol(0);
        assertThat(actualMeans).contains(SynMatrix.ones(1, 100).applyInPlace(x -> x * 3).getRow(0), precision);
    }

}