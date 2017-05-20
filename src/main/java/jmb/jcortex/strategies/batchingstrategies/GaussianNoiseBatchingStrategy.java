/*
 * James Brundege
 * Date: 2017-05-16
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.batchingstrategies;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.data.SynMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.Random;

/**
 *
 */
public class GaussianNoiseBatchingStrategy implements BatchingStrategy {

    private int batchSize;
    private double percentRandomFeatures;
    private SynMatrix featureMeans;
    private SynMatrix featureStds;
    private Random random = new Random();

    public GaussianNoiseBatchingStrategy(int batchSize, double percentRandomFeatures, DataSet trainingSet) {
        this.batchSize = batchSize;
        this.percentRandomFeatures = percentRandomFeatures;
        this.featureMeans = trainingSet.getFeatures().getColMeans();
        this.featureStds = calcSD(trainingSet.getFeatures());
    }

    private SynMatrix calcSD(SynMatrix features) {
        double[] stds = features.getStreamOfCols()
                .map(col -> new DescriptiveStatistics(col).getStandardDeviation())
                .mapToDouble(Double::doubleValue)
                .toArray();
        return new SynMatrix(stds);
    }

    @Override
    public BatchedDataSet getBatchedDataSet(DataSet trainingSet) {
        DataSet shuffledTrainingSet = trainingSet.shuffleRows();
        SynMatrix features = shuffledTrainingSet.getFeatures().copy();
        for (int rowNum = 0; rowNum < features.numRows(); rowNum++) {
            for (int colNum = 0; colNum < features.numCols(); colNum++) {
                if (random.nextDouble() < percentRandomFeatures) {
                    double mean = featureMeans.get(colNum);
                    double std = featureStds.get(colNum);
                    double value = random.nextGaussian() * std + mean;
                    features.set(rowNum, colNum, value);
                }
            }
        }
        DataSet dataSet = new DataSet(features, shuffledTrainingSet.getLabels());
        return new BatchedDataSet(dataSet, batchSize);
    }

    public void setRandom(Random random) {
        this.random = random;
    }
}