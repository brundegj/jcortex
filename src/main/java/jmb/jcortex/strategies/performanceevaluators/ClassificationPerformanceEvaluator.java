/*
 * James Brundege
 * Date: 2017-04-11
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.performanceevaluators;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import jmb.jcortex.neuralnet.NeuralNet;

import java.util.Arrays;
import java.util.List;

public class ClassificationPerformanceEvaluator implements PerformanceEvaluator {

    @Override
    public double getError(NeuralNet neuralNet, DataSet dataSet) {
        SynMatrix output = neuralNet.analyzeData(dataSet);
        SynMatrix answers = convertTo01Answers(output);
        return getPercentWrong(answers, dataSet.getLabels());
    }

    private SynMatrix convertTo01Answers(SynMatrix output) {
        SynMatrix answers = new SynMatrix(output.numRows(), output.numCols());
        for (int rowNum = 0; rowNum < output.numRows(); rowNum++) {
            double highest = 0;
            int highestCol = -1;
            double[] row = output.getRow(rowNum);
            for (int colNum = 0; colNum < row.length; colNum++) {
                if (row[colNum] > highest) {
                    highest = row[colNum];
                    highestCol = colNum;
                }
            }
            answers.set(rowNum, highestCol, 1.0);
        }
        return answers;
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
