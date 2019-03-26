package experiments ;
import nl.tue.s2id90.dl.experiment.Experiment ;
import java.io.IOException ;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

public class FunctionExperiment extends Experiment {
    // ( hyper ) parameters
    int batchSize = 32;
    // . . .
    public void go( ) throws IOException {
        // read input and pr int some informat ion on the data
        InputReader reader = GenerateFunctionData .THREE_VALUED_FUNCTION
        ( batchSize ) ;
        System.out.println ("Reader info:\n" + reader.toString( ) ) ;
    }
    
    public static void main ( String[ ] args ) throws IOException {
        new FunctionExperiment( ).go( );
    }
}
