package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;

public class MeanSubtraction implements DataTransform {
    Double mean = 0.0;
    
    @Override 
    public void fit ( List <TensorPair > data ) {
        if ( data.isEmpty()) {
            throw new IllegalArgumentException ( "Empty dataset" ) ;
        }
        
        for ( TensorPair pair : data ) {
            mean = mean + pair.model_input.getValues().meanNumber().doubleValue();
        }
        
        mean = mean / data.size();
        System.out.println("mean: " + mean);
    }
    @Override 
    public void transform ( List <TensorPair > data ) {
       for ( TensorPair pair : data ) {
            pair.model_input.getValues().addi((mean * -1.0));
        }
    }
}
