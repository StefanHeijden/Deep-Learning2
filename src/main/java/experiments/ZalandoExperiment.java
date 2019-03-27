/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import nl.tue.s2id90.dl.input.MNISTReader;
import java.io.IOException ;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.javafx.ShowCase;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.javafx.FXGUI;

public class ZalandoExperiment extends GUIExperiment {
    // ( hyper ) parameters
    int batchSize = 32;
    // The parameter epochs is the number of epochs that a
    // training takes. In an epoch all the training samples are presented
    // once to the neural network.
    int epochs = 5; 
    // Parameter for the gradient descent optimization method.
    double learningRate = 0.01;
    
    // normal parameters
    // the number of neurons of the new layer
    int imageSize = 28;
    InputReader reader;
    int inputs ;
    int outputs;
        
    String[ ] labels = {"T'shirt/top" , "Trouser" , "Pullover" , "Dress" , "Coat" ,
    "Sandal" , "Shirt" , "Sneaker" , "Bag" , "Ankle boot" } ;
    
    ShowCase showCase = new ShowCase ( i -> labels[i]) ;
    
    public void go( ) throws IOException {
        reader = MNISTReader.fashion( batchSize ) ;
        inputs = reader.getInputShape().getNeuronCount();
        outputs = reader.getOutputShape().getNeuronCount();
        
        System.out.println ("Inputs: " + inputs ) ;
        System.out.println ("Outputs: " + outputs ) ;
        
        FXGUI.getSingleton().addTab( "show case", showCase.getNode( ));
        showCase.setItems(reader.getValidationData(100));
        
        Model model = createModel ( inputs , outputs ) ;
        model. initialize (new Gaussian());
        // Training: create and configure SGD && train model
        Optimizer sgd = SGD.builder ()
            .model (model )
            .learningRate( learningRate )
            .validator(new Classification())
            .learningRate( learningRate )
            .build(); 
        trainModel(sgd ,reader ,epochs ,0);
       
        // read input and print some informat ion on the data
        System.out.println ("Reader info:\n" + reader.toString( ) ) ;
        reader.getValidationData(1).forEach(System.out::println);
    }
    
    public static void main ( String[ ] args ) throws IOException {
        new ZalandoExperiment( ).go( );
    }
    
    Model createModel(int inputs , int outputs ) {
        TensorShape image = new TensorShape ( imageSize, imageSize, 1);
        Model model = new Model(new InputLayer("In", image , true ) ) ;
        // add flattenlayer after input layer
        model.addLayer (new Flatten ( "Flatten " , image)) ;
        // add output layer
        model.addLayer (new OutputSoftmax ( "Out " , new TensorShape(image.getNeuronCount()) , outputs , new CrossEntropy( ))) ;
        return model;
    }
    
    @Override
    public void onEpochFinished( Optimizer sgd , int epoch ) {
        super.onEpochFinished( sgd ,epoch ) ;
        showCase . update ( sgd . getModel ( ) ) ;
    }
}