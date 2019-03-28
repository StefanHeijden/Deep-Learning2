package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;

public class MeanSubtraction implements DataTransform {
    Double mean ;
    
    @Override 
    public void fit ( List <TensorPair > data ) {
        if ( data.isEmpty()) {
            throw new IllegalArgumentException ( "Empty dataset" ) ;
        }
        for ( TensorPair pair : data ) {
            //. . .
        }
        // . . .
    }
    @Override 
    public void transform ( List <TensorPair > data ) {
        // To do
    }
}
