/*
 * James Brundege
 * Date: 2017-04-22
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.trainers;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;
import jmb.jcortex.strategies.batchingstrategies.BatchedDataSet;
import jmb.jcortex.strategies.batchingstrategies.BatchingStrategy;
import jmb.jcortex.strategies.haltingstrategies.HaltingStrategy;
import jmb.jcortex.strategies.optimizationstrategies.OptimizationStrategy;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class GradientDescentTrainerTest {

    @Mock
    private BatchingStrategy batchingStrategy;
    @Mock
    private OptimizationStrategy optimizationStrategy;
    @Mock
    private HaltingStrategy haltingStrategy;
    @Mock
    private DeltaCalculator deltaCalculator;
    @Mock
    private GradientCalculator gradientCalculator;

    @Mock
    private NeuralNet startingNeuralNet;
    @Mock
    private NeuralNet bestNeuralNet;
    @Mock
    private DataSet trainingSet;
    @Mock
    private BatchedDataSet batch;

    private GradientDescentTrainer gradientDescentTrainer;

    @Before
    public void setup() {
        gradientDescentTrainer = new GradientDescentTrainer(batchingStrategy, optimizationStrategy, haltingStrategy);
        gradientDescentTrainer.setDeltaCalculator(deltaCalculator);
        gradientDescentTrainer.setGradientCalculator(gradientCalculator);
    }


    @Test
    public void train_ReturnsBestNeuralNetFromHaltingStrategy() {
        when(haltingStrategy.shouldHalt()).thenReturn(true);
        when(haltingStrategy.getBestNeuralNet()).thenReturn(bestNeuralNet);

        NeuralNet trained = gradientDescentTrainer.train(startingNeuralNet, trainingSet);

        assertThat(trained).isSameAs(bestNeuralNet);
        verifyZeroInteractions(startingNeuralNet);
        verifyZeroInteractions(batchingStrategy);
        verifyZeroInteractions(optimizationStrategy);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void train_DoesBackPropagationAndUpdatesWeights() {
        List<SynMatrix> layers = Arrays.asList(
                new SynMatrix(new double[][]{
                        {0.1, -0.2},
                        {-0.1, 0.2},
                        {0.15, -0.15}
                }),
                new SynMatrix(new double[][]{
                        {-0.1, 0.2},
                        {0.15, -0.15}
                })
        );
        List<SynMatrix> nodeValues = mock(List.class);
        SynMatrix labels = mock(SynMatrix.class);
        List<SynMatrix> deltas = mock(List.class);
        List<SynMatrix> gradients = mock(List.class);
        List<SynMatrix> corrections = Arrays.asList(
                new SynMatrix(new double[][]{
                        {0.2, -0.1},
                        {-0.2, 0.1},
                        {0.3, -0.3}
                }),
                new SynMatrix(new double[][]{
                        {-0.2, 0.1},
                        {0.3, -0.3}
                })
        );

        // The update should be layers - corrections
        List<SynMatrix> expectedLayers = Arrays.asList(
                new SynMatrix(new double[][]{
                        {-0.1, -0.1},
                        {0.1, 0.1},
                        {-0.15, 0.15}
                }),
                new SynMatrix(new double[][]{
                        {0.1, 0.1},
                        {-0.15, 0.15}
                })
        );

        when(haltingStrategy.shouldHalt()).thenReturn(false, true); // train one iteration
        when(haltingStrategy.getBestNeuralNet()).thenReturn(bestNeuralNet);
        when(batchingStrategy.getBatchedDataSet(trainingSet)).thenReturn(batch);
        when(batch.hasNext()).thenReturn(true, false);  // one batch
        when(batch.getNextBatch()).thenReturn(trainingSet);
        when(trainingSet.getLabels()).thenReturn(labels);
        when(startingNeuralNet.trainForward(trainingSet)).thenReturn(nodeValues);
        when(startingNeuralNet.getLayers()).thenReturn(layers);
        when(deltaCalculator.calcDeltas(nodeValues, labels, startingNeuralNet)).thenReturn(deltas);
        when(gradientCalculator.calcGradients(deltas, nodeValues)).thenReturn(gradients);
        when(optimizationStrategy.calcCorrections(gradients)).thenReturn(corrections);
        ArgumentCaptor<List<SynMatrix>> layerCaptor = ArgumentCaptor.forClass(List.class);

        NeuralNet trained = gradientDescentTrainer.train(startingNeuralNet, trainingSet);

        assertThat(trained).isSameAs(bestNeuralNet);
        verify(haltingStrategy, times(2)).evaluate(startingNeuralNet);
        verify(startingNeuralNet).setLayers(layerCaptor.capture());
        List<SynMatrix> actualLayers = layerCaptor.getValue();
        assertThat(actualLayers).isEqualTo(expectedLayers);
    }

}