import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.core.transfer.Tanh;
import org.neuroph.util.ConnectionFactory;

/**
 *
 * @author lily
 */


class MyNeuralNetwork{

    NeuralNetwork ann;
  
    public MyNeuralNetwork()
    {
        //Se crea la red
    //   ann = new NeuralNetwork(TransferFunctionType.SIGMOID, 1, 20, 1);
       ann = new NeuralNetwork();
       
       //Se crean las capas con sus respectivas neuronas
       Layer capaEntrada = new Layer();
       capaEntrada.addNeuron(new Neuron());
       capaEntrada.addNeuron(new Neuron());
       capaEntrada.addNeuron(new Neuron());       
       capaEntrada.addNeuron(new Neuron());
       for(int i=0; i < capaEntrada.getNeuronsCount(); i++)
       {
           capaEntrada.getNeuronAt(i).setTransferFunction(new Tanh()/*new Sigmoid()*/);
       }
       
       Layer capaOculta = new Layer();
       capaOculta.addNeuron(new Neuron());
       capaOculta.addNeuron(new Neuron());
       capaOculta.addNeuron(new Neuron());
       capaOculta.addNeuron(new Neuron());
       capaOculta.addNeuron(new Neuron());
       capaOculta.addNeuron(new Neuron());
       for(int i=0; i < capaOculta.getNeuronsCount(); i++)
       {
           capaOculta.getNeuronAt(i).setTransferFunction(new Tanh()/*new Sigmoid()*/);
       }
       
       Layer capaSalida = new Layer();
       capaSalida.addNeuron(new Neuron());
       capaSalida.addNeuron(new Neuron());
       capaSalida.addNeuron(new Neuron());
       capaSalida.addNeuron(new Neuron());
       capaSalida.addNeuron(new Neuron());     
       for(int i=0; i < capaSalida.getNeuronsCount(); i++)
       {
           capaSalida.getNeuronAt(i).setTransferFunction(new Tanh()/*new Sigmoid()*/);
       }
        
       
//Se agregan a la red neuronal
       ann.addLayer(0,capaEntrada);
       ann.addLayer(1,capaOculta);
       ann.addLayer(2, capaSalida);
       //Se genera la conexión entre capas
       ConnectionFactory.fullConnect(capaEntrada, capaOculta, 0);
       ConnectionFactory.fullConnect(capaOculta, capaSalida, 0);
       //ConnectionFactory.fullConnect(capaEntrada, capaSalida, 0);
       //Se indican la capa de entrada y salida
       ann.setInputNeurons(capaEntrada.getNeurons());
       ann.setOutputNeurons(capaSalida.getNeurons());
        
       //Por defecto, la función de activación es de step, cambiamos a tangente hiperbólica
    /*   for(int i = 0; i < ann.getLayersCount(); i++)
       {
           for(int j = 0; j < ann.getLayerAt(i).getNeuronsCount(); j++)
           {
               ann.getLayerAt(i).getNeuronAt(j).setTransferFunction(new Tanh());
               System.out.println("NEURONA: " + (ann.getLayerAt(i).getNeuronAt(j).getTransferFunction().getOutput(-0.29)));
           }
       }     */ 
    }

    public void setMyWeights(float[] w){
      double[] weights = new double[w.length];
      for(int i = 0; i < w.length; i++){
          weights[i] = (double)w[i];
      }      
      this.ann.setWeights(weights);
    }

    public void setMyInput(double[] i, double risk){
      double[] norm = new double[i.length+1]; 
      for(int x  = 0; x < i.length; x++){
        norm[x] = i[x];
      }
      norm[i.length] = risk;
      this.ann.setInput(NormalizeInput(norm));
  }

  float[] getAngles() {
        //calcular los valores de la red
        ann.calculate();
        //Obtener los valores de la red        
        float[] result = NormalizeOutput(ann.getOutput());        
        return result;
    }

  public double[] NormalizeInput(double[] i){
    double[] temp = new double[i.length];
    double vMax = 700;
    for (int x = 0; x < i.length-1; x++) {
      if(i[x] < 0){
        i[x] = i[x] * -1;
      }      
      temp[x] = i[x]/vMax;
      //System.out.println("Input " + x + ": " + temp[x]);
    }
    temp[i.length-1] = i[i.length-1];
    //System.out.println("Input " + (i.length-1) + ": " + temp[i.length-1]);
    return temp;
  }

  public float[] NormalizeOutput(double[] o){
    float[] output = new float[o.length];
    for (int x = 0; x < o.length; x++) {
      output[x] = (float)((o[x] * (Math.PI*2)));
      if(output[x]>0){
        output[x] =  (float)(output[x] - (Math.PI));
      }
      else{
         output[x] =  (float)(output[x] + (Math.PI));
      }
      //output[x] = (float)(o[x]);
 //     System.out.println("Output " + x + ": " + (o[x])+", " +output[x]);
    }
    return output;
  }
}
