/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author stefa
 */
public class MyGradientDescentVariant implements UpdateFunction{
    INDArray update;
    @Override
    public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        // on the first call of this method , create update vector .
        if ( update==null ) update = gradient.dup('f').assign(0);
        
        double factor = -1.0 * ( learningRate/batchSize ) ;
        // array <-- array + factor * gradient
        Nd4j.getBlasWrapper().level1().axpy(array.length(), factor, gradient, array);
    }
    
}
