/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.function.Supplier;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author 20173998
 */
public class L2Decay implements UpdateFunction {
    double decay;
    UpdateFunction f;
    
    public L2Decay(Supplier<UpdateFunction> supplier, double decay) {
        this.decay = decay;
        this.f = supplier.get();
    }
    
    @Override
    public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        double factor = -(learningRate/batchSize);
        Nd4j.getBlasWrapper().level1().axpy( array.length(), factor, gradient, array );
        if (!isBias) {
           array.subi(array.mul(decay)) ;
        }
    }
}