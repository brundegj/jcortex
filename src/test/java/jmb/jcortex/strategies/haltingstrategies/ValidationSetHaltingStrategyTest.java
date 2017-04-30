/*
 * James Brundege
 * Date: 2017-04-17
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.haltingstrategies;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.strategies.performanceevaluators.PerformanceEvaluator;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class ValidationSetHaltingStrategyTest {

    @Mock
    private PerformanceEvaluator performanceEvaluator;
    @Mock
    private DataSet trainingSet;
    @Mock
    private DataSet validationSet;
    @Mock
    private NeuralNet neuralNet;
    @Mock
    private NeuralNet bestNeuralNet;

    private ValidationSetHaltingStrategy validationSetHaltingStrategy;

    @Before
    public void setup() {
        validationSetHaltingStrategy = new ValidationSetHaltingStrategy(trainingSet, validationSet, performanceEvaluator, 2);
    }

    @Test
    public void testHaltingStrategy() {
        when(performanceEvaluator.getError(neuralNet, trainingSet)).thenReturn(0.5);
        when(performanceEvaluator.getError(neuralNet, validationSet)).thenReturn(0.6);
        validationSetHaltingStrategy.evaluate(neuralNet);
        assertThat(validationSetHaltingStrategy.shouldHalt()).isFalse();

        // A new best iteration on the validation set
        when(bestNeuralNet.copy()).thenReturn(bestNeuralNet);
        when(performanceEvaluator.getError(bestNeuralNet, trainingSet)).thenReturn(0.6);
        when(performanceEvaluator.getError(bestNeuralNet, validationSet)).thenReturn(0.5);
        validationSetHaltingStrategy.evaluate(bestNeuralNet);
        assertThat(validationSetHaltingStrategy.shouldHalt()).isFalse();

        // Not as good #1
        when(performanceEvaluator.getError(neuralNet, trainingSet)).thenReturn(0.4);
        when(performanceEvaluator.getError(neuralNet, validationSet)).thenReturn(0.55);
        validationSetHaltingStrategy.evaluate(neuralNet);
        assertThat(validationSetHaltingStrategy.shouldHalt()).isFalse();

        // Not as good #2
        when(performanceEvaluator.getError(neuralNet, trainingSet)).thenReturn(0.4);
        when(performanceEvaluator.getError(neuralNet, validationSet)).thenReturn(0.55);
        validationSetHaltingStrategy.evaluate(neuralNet);
        assertThat(validationSetHaltingStrategy.shouldHalt()).isTrue();

        assertThat(validationSetHaltingStrategy.getBestNeuralNet()).isEqualTo(bestNeuralNet);
    }

}