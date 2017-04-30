package jmb.jcortex.trainers;

import jmb.jcortex.data.DataSet;
import jmb.jcortex.neuralnet.NeuralNet;

/**
 * James Brundege
 * Date: 2017-04-09
 * MIT license: https://opensource.org/licenses/MIT
 */
public interface SupervisedTrainer {

    NeuralNet train(NeuralNet neuralNet, DataSet trainingSet);

}
