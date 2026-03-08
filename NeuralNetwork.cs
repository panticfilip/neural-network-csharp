using System;
using System.Diagnostics;
using System.IO;

namespace NeuralNetworkLibrary
{
    public class Sample
    {
        public double [] input;
        public double[] output;
        public Sample(double[] inp, double[] oup)
        {
            input = inp;
            output = oup;
        }
    }
    public class Neuron
    {
        public double[] weights;
        public double bias;
        public double activation;
        public double[] weightGradients;
        public double biasGradient;
        public Neuron(int inputSize)
        {
            weights = new double[inputSize];
            weightGradients = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                weights[i] = (NeuralNetwork.rand.NextDouble() * 0.2) - 0.1;
                weightGradients[i] = 0;
            }
            bias = (NeuralNetwork.rand.NextDouble() * 0.2) - 0.1;
        }
    }
    public class NeuralNetwork
    {
        public static Random rand = new Random();
        public Neuron[][] layers;
        public double learningRate;
        public int batchSize;
        public double correct;
        public double[] accuracy;
        string path = "";
        public int epochs;
        public NeuralNetwork(int[] b, int inputSize, double lr, int batchSz, int ep)
        {
            learningRate = lr;
            batchSize = batchSz;
            epochs = ep;
            layers = new Neuron[b.Length][];
            accuracy = new double[epochs];
            for (int i = 0; i < b.Length; i++)
            {
                layers[i] = new Neuron[b[i]];
                for (int j = 0; j < b[i]; j++)
                {
                    if (i == 0)
                    {
                        layers[i][j] = new Neuron(inputSize);
                    }
                    else
                    {
                        layers[i][j] = new Neuron(b[i - 1]);
                    }
                }
            }
        }
        public void AddPath(string pth)
        {
            path = pth;
        }
        static double ReLU(double a)
        {
            return Math.Max(0, a);
        }
        public void Softmax(ref double[] a)
        {
            double max = double.NegativeInfinity;
            for (int i = 0; i < a.Length; i++)
                if (a[i] > max) max = a[i];
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = Math.Exp(a[i] - max);
                sum += a[i];
            }
            for (int i = 0; i < a.Length; i++)
            {
                a[i] /= sum;
            }
        }
        public virtual void FeedForward(double[] input)
        {
            double[] currentInput;
            double[] nextInput = input;
            double[] outputLayer = new double[layers[layers.Length - 1].Length];
            for (int layer = 0; layer < layers.Length; layer++)
            {
                currentInput = nextInput;
                nextInput = new double[layers[layer].Length];
                for (int i = 0; i < layers[layer].Length; i++)
                {
                    Neuron neuron = layers[layer][i];
                    double sum = 0;
                    for (int j = 0; j < neuron.weights.Length; j++)
                    {
                        sum += neuron.weights[j] * currentInput[j];
                    }
                    sum += neuron.bias;
                    if (layer == layers.Length - 1)
                    {
                        layers[layer][i].activation = sum;
                    }
                    else
                    {
                        neuron.activation = ReLU(sum);
                    }
                    nextInput[i] = neuron.activation;
                }
            }
        }
        public virtual double LastLayerDelta(double a, double output) { return 0; }
        public virtual void Backpropagation(double[] expectedOutput, double[] input)
        {
            double[][] previousDelta = new double[layers.Length][];
            for (int layer = layers.Length - 1; layer >= 0; layer--)
            {
                previousDelta[layer] = new double[layers[layer].Length];
                for (int i = 0; i < layers[layer].Length; i++)
                {
                    Neuron neuron = layers[layer][i];
                    double delta = 0;
                    if (layer == layers.Length - 1)
                    {
                        delta = LastLayerDelta(neuron.activation, expectedOutput[i]);
                    }
                    else
                    {
                        for (int k = 0; k < layers[layer + 1].Length; k++)
                        {
                            delta += previousDelta[layer + 1][k] * layers[layer + 1][k].weights[i];
                        }
                        if (neuron.activation <= 0)
                        {
                            delta = 0;
                        }
                    }
                    for (int j = 0; j < neuron.weights.Length; j++)
                    {
                        if (layer == 0)
                        {
                            neuron.weightGradients[j] += delta * (double)input[j];
                        }
                        else
                        {
                            neuron.weightGradients[j] += delta * layers[layer - 1][j].activation;
                        }
                    }
                    previousDelta[layer][i] = delta;
                    neuron.biasGradient += delta;
                }
            }
        }
        public void UpdateWeights()
        {
            for (int layer = 0; layer < layers.Length; layer++)
            {
                for (int i = 0; i < layers[layer].Length; i++)
                {
                    Neuron neuron = layers[layer][i];
                    for (int j = 0; j < neuron.weights.Length; j++)
                    {
                        neuron.weights[j] -= (learningRate / batchSize) * neuron.weightGradients[j];
                        neuron.weightGradients[j] = 0;
                    }
                    neuron.bias -= (learningRate / batchSize) * neuron.biasGradient;
                    neuron.biasGradient = 0;
                }
            }
        }
        public void SaveWeights()
        {
            StreamWriter sw = new StreamWriter(path);

            foreach (var layer in layers)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron.weights)
                    {
                        sw.Write("{0};", weight);
                    }
                    sw.WriteLine(neuron.bias);
                }
            }
            sw.Close();
        }
        static void ShuffleArray<T>(T[] a)
        {
            for (int i = a.Length - 1; i > 0; i--)
            {
                int j = rand.Next(0, i + 1);
                T temp = a[i];
                a[i] = a[j];
                a[j] = temp;
            }
        }
        public void LoadWeights()
        {
            if (File.Exists(path))
            {
                StreamReader sr = new StreamReader(path);
                string s = sr.ReadToEnd();
                sr.Close();
                string[] lines = s.Split('\n');
                int lineNum = 0;
                if (s.Length != 0)
                {
                    for (int layer = 0; layer < layers.Length; layer++)
                    {
                        for (int i = 0; i < layers[layer].Length; i++)
                        {
                            string[] line = lines[lineNum].Split(';');
                            lineNum++;
                            if (layers[layer][i].weights.Length + 1 != line.Length)
                            {
                                throw new Exception("The neural network architecture is not compatible with the file structure.");
                            }
                            else
                            {
                                for (int j = 0; j < layers[layer][i].weights.Length; j++)
                                {
                                    layers[layer][i].weights[j] = double.Parse(line[j]);
                                }
                                layers[layer][i].bias = double.Parse(line[line.Length - 1]);
                            }
                        }
                    }
                }
            }
        }
        public void Train(Sample[] samples, bool Write)
        {
            LoadWeights();
            for (int i = 0; i < epochs; i++)
            {
                ShuffleArray(samples);
                int num = 1;
                correct = 0;
                foreach (var sample in samples)
                {
                    FeedForward(sample.input);
                    Backpropagation(sample.output, sample.input);
                    if (num % batchSize == 0 || num == samples.Length) UpdateWeights();
                    num++;
                }
                accuracy[i] = correct / (double)samples.Length;
                if (Write) Console.WriteLine("Epoch {0} accuracy: {1:F2}%", i + 1, accuracy[i] * 100);
            }
            if (File.Exists(path)) SaveWeights();
        }
    }
    public class ClassificationNetwork : NeuralNetwork
    {
        public ClassificationNetwork(int[] b, int inputSize, double lr, int batchSz, int ep) : base(b, inputSize, lr, batchSz, ep) { }
        public int Prediction()
        {
            int predicted = 0;
            double maxAct = layers[layers.Length - 1][0].activation;
            for (int i = 1; i < layers[layers.Length - 1].Length; i++)
            {
                if (layers[layers.Length - 1][i].activation > maxAct)
                {
                    maxAct = layers[layers.Length - 1][i].activation;
                    predicted = i;
                }
            }
            return predicted;
        }
        public override double LastLayerDelta(double a, double output)
        {
            return a - output;
        }
        public override void Backpropagation(double[] expectedOutput, double[] input)
        {
            base.Backpropagation(expectedOutput, input);
            int actual = 0;
            for (int i = 0; i < expectedOutput.Length; i++)
            {
                if (expectedOutput[i] == 1)
                {
                    actual = i;
                    break;
                }
            }
            if (Prediction() == actual)
            {
                correct++;
            }
        }
        public override void FeedForward(double[] input)
        {
            base.FeedForward(input);
            double[] a = new double[layers[layers.Length - 1].Length];
            for (int i = 0; i < layers[layers.Length-1].Length; i++)
            {
                a[i] = layers[layers.Length - 1][i].activation;
            }
            Softmax(ref a);
            for (int i = 0; i < layers[layers.Length-1].Length; i++)
            {
                layers[layers.Length - 1][i].activation = a[i];
            }
        }
        public int TestNetwork(double[] input)
        {
            FeedForward(input);
            return Prediction();
        }
    }
    public class RegressionNetwork : NeuralNetwork
    {
        public double tolerance = 0.001;
        public RegressionNetwork(int[] b, int inputSize, double lr, int batchSz, int ep) : base(b, inputSize, lr, batchSz, ep) { }
        public RegressionNetwork(int[] b, int inputSize, double lr, int batchSz, int ep, double tlr) : base(b, inputSize, lr, batchSz, ep)
        {
            tolerance = tlr;
        }
        public double[] Prediction()
        {
            double[] a = new double[layers[layers.Length - 1].Length];
            for (int i = 0; i < layers[layers.Length - 1].Length; i++)
            {
                a[i] = layers[layers.Length - 1][i].activation;
            }
            return a;
        }
        public override double LastLayerDelta(double a, double output)
        {
            return (a - output) * (a * (1 - a));
        }
        public override void Backpropagation(double[] expectedOutput, double[] input)
        {
            base.Backpropagation(expectedOutput, input);
            bool Correct = true;
            for (int i = 0; i < expectedOutput.Length; i++)
            {
                if (Math.Abs(expectedOutput[i] - Prediction()[i]) > tolerance) { Correct = false; break; }
            }
            if (Correct) correct++;
        }
        public override void FeedForward(double[] input)
        {
            base.FeedForward(input);
            for(int i =0; i<layers[layers.Length - 1].Length; i++)
            {
                layers[layers.Length - 1][i].activation = 1.0/(1.0+Math.Exp(-layers[layers.Length - 1][i].activation));
            }
        }
        public double[] TestNetwork(double[] input)
        {
            FeedForward(input);
            return Prediction();
        }
    }
}