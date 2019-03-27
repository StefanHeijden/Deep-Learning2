package experiments ;
import nl.tue.s2id90.dl.experiment.Experiment ;
import java.io.IOException ;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.experiment.GUIExperiment;

public class FunctionExperiment extends GUIExperiment {
    // ( hyper ) parameters
    int batchSize = 32;
    // The parameter epochs is the number of epochs that a
    // training takes. In an epoch all the training samples are presented
    // once to the neural network.
    int epochs = 10; 
    // Parameter for the gradient descent optimization method.
    double learningRate = 0.01;
    
    // normal parameters
    // the number of neurons of the new layer
    int n = 5;
    InputReader reader = GenerateFunctionData .THREE_VALUED_FUNCTION( batchSize );
    int inputs = reader.getInputShape().getNeuronCount();
    int outputs = reader.getOutputShape().getNeuronCount();
        
    public void go( ) throws IOException {
        System.out.println ("Inputs: " + inputs ) ;
         System.out.println ("Outputs: " + outputs ) ;
         
        Model model = createModel ( inputs , outputs ) ;
        model. initialize (new Gaussian());
        // Training : create and configure SGD && trainmodel
        Optimizer sgd = SGD.builder ()
            .model (model )
            .validator(new Regression() )
            .learningRate( learningRate )
            .build();
            
        trainModel(sgd ,reader ,epochs ,0) ;
       
        // read input and print some informat ion on the data
        System.out.println ("Reader info:\n" + reader.toString( ) ) ;
        reader.getValidationData(20).forEach(System.out::println);
    }
    
    public static void main ( String[ ] args ) throws IOException {
        new FunctionExperiment( ).go( );
    }
    
    Model createModel(int inputs , int outputs ) {
        Model model = new Model(new InputLayer("In", new TensorShape(inputs ) , true ) ) ;
        model.addLayer(new FullyConnected ( "fc1" , new TensorShape(inputs) , n , new RELU( )) ) ;
        model.addLayer(new SimpleOutput("Out ", new TensorShape(n) , outputs , new MSE() , true ) ) ;
        return model;
    }
}