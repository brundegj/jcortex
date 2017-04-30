/*
 * James Brundege
 * Date: 2017-04-29
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 *
 */
public class ThresholdPerformanceEvaluator implements PerformanceEvaluator {

    private double threshold;

    public ThresholdPerformanceEvaluator(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public double getError(NeuralNet neuralNet, DataSet dataSet) {
        SynMatrix output = neuralNet.analyzeData(dataSet);
        SynMatrix answers = convertTo01Answers(output);
        return getPercentWrong(answers, dataSet.getLabels());
    }

    private SynMatrix convertTo01Answers(SynMatrix output) {
        return output.apply((DoubleUnaryOperator) x -> x >= threshold ? 1 : 0);
    }

    private double getPercentWrong(SynMatrix answers, SynMatrix labels) {
        List<double[]> answerRows = answers.getRows();
        List<double[]> labelRows = labels.getRows();
        int numWrong = 0;
        for (int i = 0; i < answerRows.size(); i++) {
             if (!Arrays.equals(answerRows.get(i), labelRows.get(i))) {
                 numWrong++;
             }
        }
        return (double)numWrong/(double)answerRows.size();
    }

}

