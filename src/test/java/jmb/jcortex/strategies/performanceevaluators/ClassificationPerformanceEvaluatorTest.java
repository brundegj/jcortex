/*
 * James Brundege
 * Date: 2017-04-19
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;
import org.assertj.core.data.Offset;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class ClassificationPerformanceEvaluatorTest {
    private Offset<Double> precision = Offset.offset(0.000000000000001);

    @Mock
    private NeuralNet neuralNet;
    @Mock
    private DataSet dataSet;

    @Test
    public void test() {
        double[][] outputValues = new double[][] {
                {0.35, 1.2, 1.7, 0.98},
                {1.1, 0.01, 0.1, 1.05},
                {0.002, 0.0021, 0.001, 0.0003},
                {2, 3, 4, 5}
        };
        SynMatrix output = new SynMatrix(outputValues);
        double[][] labelValues = new double[][] {
                {0, 0, 1, 0},
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}    // last one is wrong above (position 3 doesn't have the highest value
        };
        SynMatrix labels = new SynMatrix(labelValues);

        when(neuralNet.analyzeData(dataSet)).thenReturn(output);
        when(dataSet.getLabels()).thenReturn(labels);

        ClassificationPerformanceEvaluator performanceEvaluator = new ClassificationPerformanceEvaluator();
        double error = performanceEvaluator.getError(neuralNet, dataSet);

        assertThat(error).isEqualTo(0.25, precision);
    }

}