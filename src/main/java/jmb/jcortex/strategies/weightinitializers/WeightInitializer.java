/*
 * James Brundege
 * Date: 2017-04-12
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.strategies.weightinitializers;

import jmb.jcortex.data.SynMatrix;

public interface WeightInitializer {

    SynMatrix initialize(SynMatrix matrix);

}
