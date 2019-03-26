package experiments ;
import nl.tue.s2id90.dl.experiment.Experiment ;
import java.io.IOException ;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.validate.Regression;

public class FunctionExperiment extends Experiment {
    // ( hyper ) parameters
    int epochs=10; 
    double learningRate = 0.01;
    
    // normal parameters
    int batchSize = 32;
    InputReader reader = GenerateFunctionData .THREE_VALUED_FUNCTION( batchSize ) ;
    int inputs = reader.getInputShape().getNeuronCount();
    int outputs = reader.getOutputShape().getNeuronCount();
        
    public void go( ) throws IOException {
        Model model = createModel ( inputs , outputs ) ;
        model. initialize (new Gaussian());
        // Training : c r e a t e and conf igure SGD && t r a i n model
        Optimizer sgd = SGD.builder ()
            .model (model )
            .validator(new Regression() )
            .learningRate( learningRate )
            .build();
            
        trainModel(sgd ,reader ,epochs ,0) ;
       
        // read input and pr int some informat ion on the data
        System.out.println ("Reader info:\n" + reader.toString( ) ) ;
        reader.getValidationData(10).forEach(System.out::println);
    }
    
    public static void main ( String[ ] args ) throws IOException {
        new FunctionExperiment( ).go( );
    }
    
    Model createModel(int inputs , int outputs ) {
        Model model = new Model(new InputLayer("In", new TensorShape(inputs ) , true ) ) ;
        model.addLayer(new SimpleOutput("Out ", new TensorShape(inputs ) , outputs , new MSE() , true ) ) ;
        return model;
    }
}