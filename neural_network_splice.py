# Bioinformatic methods
# Neura Network Splice algorithm. Implementation and testing

import numpy as np
import pandas as pd
import re
import time
import sys
import string
from numpy.distutils.log import good

def read_data(file_name):
    return np.loadtxt(file_name, dtype = object, skiprows = 1)

def extract_target_values(input_data):
    return [int(value) for value in input_data[::2]]

def extract_dna_data(input_data):
    return [str(value) for value in input_data[1::2]]

def remove_missing_values(input_data, target_vector):
    nucleotids = string.ascii_uppercase.translate(None, 'ACGT')
    nucleotids = [letter for letter in nucleotids]
    index_dna_with_missing_values = [input_data.index(dna) for dna in input_data for s in nucleotids if s in dna]
    output_data = np.delete(input_data, index_dna_with_missing_values) # delete dna sequences with missing values from data
    output_targets = np.delete(target_vector, index_dna_with_missing_values) # delete targets dna sequences of which contain missing values 
    output_targets = np.asarray(map(int, output_targets))
    return (output_data, output_targets)    
    
def convert_dna_sequence(dna_sequence):
    new_dna_seq = re.sub('A', '1000', dna_sequence)
    new_dna_seq = re.sub('C', '0100', new_dna_seq)
    new_dna_seq = re.sub('G', '0010', new_dna_seq)
    new_dna_seq = re.sub('T', '0001', new_dna_seq)
    return new_dna_seq
     
def convert_dna_data(input_data):
    converted_dna = map(convert_dna_sequence, input_data)
    return np.asarray([[int(x) for x in dna] for dna in converted_dna])

def split_data(data, target_vector):
    indices = np.random.permutation(target_vector.shape[0])
    len_training_set = int(0.7 * len(target_vector))
    training_idx, test_idx = indices[:len_training_set], indices[len_training_set:]
    training_set, test_set = data[training_idx,:], data[test_idx,:]
    training_targets, test_targets = target_vector[training_idx], target_vector[test_idx]
    return (training_set, training_targets, test_set, test_targets)

def get_preprocessed_data(file_name):  
    try:
        dataset = read_data(file_name)
    except:
        print "Problem with opening or reading the file"
        sys.exit()

    target_vector = extract_target_values(dataset)
    data = extract_dna_data(dataset)
    
    # Clean data from missing values
    data, target_vector = remove_missing_values(data, target_vector)
    data = convert_dna_data(data)
    return split_data(data, target_vector)

def initialize_weights(n_input_neurons, n_output_neurons):
    weights = 2 * np.random.random((1 + n_input_neurons, n_output_neurons)) - 1
    # return weights
    return np.transpose([0.01 * w / np.sqrt(sum(w ** 2)) for w in np.transpose(weights)])
    # return np.transpose([0.01 * w / np.mean(w) for w in np.transpose(weights)])


class NeuralNetwork():
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate):
        # the first column contains biases for each neuron in each layer
        self.input_hidden_weights = initialize_weights(n_input_neurons, n_hidden_neurons)
        self.hidden_output_weights = initialize_weights(n_hidden_neurons, n_output_neurons)          
        self.learning_rate = learning_rate
        self.mse = 0
        self.error = 0        
        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        self.specificity = 0
        self.mse_cv = 0
        self.error_cv = 0        
        self.accuracy_cv = 0
        self.recall_cv = 0
        self.precision_cv = 0
        self.specificity_cv = 0
        self.mse_valid = 0
        self.error_valid = 0        
        self.accuracy_valid = 0
        self.recall_valid = 0
        self.precision_valid = 0
        self.specificity_valid = 0
        
    def __sigmoid(self, x):
        return (1. / (1. + np.exp(-x)))
        # return np.tanh(x)
    
    def __sigmoid_prime(self, x):
        return np.exp(-x) / (np.power(1 + np.exp(-x), 2))     

    def __feedforward(self, input_neurons): # input_neurons is a vector of a dataset
        hidden_neurons = np.dot(input_neurons, self.input_hidden_weights[1:])
        hidden_neurons = self.__sigmoid(np.asarray(map(sum, zip(hidden_neurons, self.input_hidden_weights[0])))) 
        
        output_neurons = np.dot(hidden_neurons, self.hidden_output_weights[1:])
        output_neurons = self.__sigmoid(np.asarray(map(sum, zip(output_neurons, self.hidden_output_weights[0]))))        
        return (hidden_neurons, output_neurons)
    
    # Feedforward for test data
    def feedforward(self, input_neurons, input_hidden_weights, hidden_output_weights):
        self.input_hidden_weights = input_hidden_weights
        self.hidden_output_weights = hidden_output_weights
        (hidden_neurons, output_neurons) = self.__feedforward(input_neurons)
        return output_neurons
    
    # train with backpropagation    
    def __train(self, data, target_vector):
        true_neg = 0; true_pos = 0; false_neg = 0; false_pos = 0
        for i in range(len(data)):
            hidden_neurons, output_neurons = self.__feedforward(data[i])
           
            error_output_layer = output_neurons * (1 - output_neurons) * (target_vector[i] - output_neurons)                  
            error_hidden_layer = hidden_neurons * (1 - hidden_neurons) * np.dot(error_output_layer, np.transpose(self.hidden_output_weights[1:]))

            self.hidden_output_weights[1:] += [self.learning_rate * error_output_layer * neurons for neurons in self.__sigmoid(hidden_neurons)]
            self.input_hidden_weights[1:] += [self.learning_rate * error_hidden_layer * neurons for neurons in self.__sigmoid(np.asarray(data[i]))]
            
            self.hidden_output_weights[0] += self.learning_rate * error_output_layer    # Update biases
            self.input_hidden_weights[0] += self.learning_rate * error_hidden_layer     # Update biases
                        
            self.mse += np.power(target_vector[i] - output_neurons, 2)
            output_neurons = round(output_neurons)
            
            if output_neurons != target_vector[i]:
                if target_vector[i] == 1:
                    false_neg += 1
                else:
                    false_pos += 1
            else:
                if target_vector[i] == 1:
                    true_pos += 1
                else: 
                    true_neg += 1
                    
        self.mse = float(self.mse / len(data))
        self.accuracy =  (true_pos + true_neg) / float(true_pos + true_neg + false_pos + false_neg)
        self.error = 1. - self.accuracy
        self.recall = true_pos / float(true_pos + false_neg)
        self.precision = true_pos / float(true_pos + false_pos)
        self.specificity = true_neg / float(true_neg + false_pos)
        
        return output_neurons
    
    # external function for training (without CV)
    def train_epoch(self, data, target_vector, n_epoch):
        self._train_epoch(data, target_vector, n_epoch)
    
    # internal function in case of training with CV    
    def _train_epoch(self, data, target_vector, n_epoch):
        prev_prev_accuracy = self.accuracy
        prev_accuracy = self.accuracy
        learning_rate_0 = self.learning_rate
        epoch = 1
        while ((epoch <= n_epoch)):# & (not((prev_prev_accuracy > prev_accuracy) & (prev_accuracy > self.accuracy)))):
            prev_prev_accuracy = prev_accuracy
            prev_accuracy = self.accuracy
            self.__train(data, target_vector)            
            self.learning_rate = learning_rate_0 / (1. + epoch / float(len(data)))
            epoch += 1
    
    def train_epoch_with_cross_validation(self, data, target_vector, n_epoch, k_fold = 10):
        indices = np.random.permutation(target_vector.shape[0])
        len_test_set = int(1./ k_fold * len(target_vector))
               
        for i in range(0, k_fold):            
            valid_idx = indices[(len_test_set * i):(len_test_set * (i + 1))]      
            training_set, valid_set = np.delete(data, valid_idx, axis = 0), data[valid_idx]
            training_targets, valid_targets = np.delete(target_vector, valid_idx), target_vector[valid_idx]            
            
            self._train_epoch(training_set, training_targets, n_epoch)
            
            self.mse_cv += self.mse
            self.error_cv += self.error
            self.accuracy_cv += self.accuracy             
            self.recall_cv += self.recall 
            self.precision_cv += self.precision 
            self.specificity_cv += self.specificity
            
            (self.mse_valid, self.error_valid, self.accuracy_valid,  self.recall_valid, self.precision_valid, self.specificity_valid) = [x + y for x, y in zip([self.mse_valid, self.error_valid, self.accuracy_valid, self.recall_valid, self.precision_valid, self.specificity_valid], self._predict(valid_set, valid_targets))]
            # print ("Valid: ", self.error_valid, self.accuracy_valid)
            # print ("CV: ", self.error_cv, self.accuracy_cv)
            
        [self.mse_cv, self.error_cv, self.accuracy_cv,  self.recall_cv, self.precision_cv, self.specificity_cv] = np.array([self.mse_cv, self.error_cv, self.accuracy_cv,  self.recall_cv, self.precision_cv, self.specificity_cv]) / float(k_fold)
        [self.mse_valid, self.error_valid, self.accuracy_valid,  self.recall_valid, self.precision_valid, self.specificity_valid] = np.array([self.mse_valid, self.error_valid, self.accuracy_valid,  self.recall_valid, self.precision_valid, self.specificity_valid]) / float(k_fold) 
        
    def predict(self, data, target_vector):
        return self._predict(data, target_vector)
    
    def _predict(self, data, target_vector):
        true_neg = 0; true_pos = 0; false_neg = 0; false_pos = 0
        mse_test = 0
        error_test = 0
        accuracy_test = 0
        for i in range(len(data)):
            output_neurons = self.__feedforward(data[i])[1]
            mse_test += np.power(target_vector[i] - output_neurons, 2)
            output_neurons = round(output_neurons)
            if output_neurons != target_vector[i]:
                if target_vector[i] == 1:
                    false_neg += 1
                else:
                    false_pos += 1
            else:
                if target_vector[i] == 1:
                    true_pos += 1
                else: 
                    true_neg += 1
                    
        mse = float(self.mse / len(data))
        accuracy =  (true_pos + true_neg) / float(true_pos + true_neg + false_pos + false_neg)
        error = 1. - accuracy
        recall = true_pos / float(true_pos + false_neg)
        precision = true_pos / float(true_pos + false_pos)
        specificity = true_neg / float(true_neg + false_pos)
        return (mse, error, accuracy, recall, precision, specificity)
    
    def save_weights(self, filename_input_hidden_w, filename_hidden_output_w):
        pd.DataFrame(self.input_hidden_weights).to_csv(filename_input_hidden_w)
        pd.DataFrame(self.hidden_output_weights).to_csv(filename_hidden_output_w)
        
if __name__ == "__main__":        
    if (len(sys.argv) > 2):
        print "You enter too many parameters"
        sys.exit()
    if (len(sys.argv) < 2):
        print "Please enter the file name"
        sys.exit()
    else:
        file_name = sys.argv[1]
    
    if "Donor" in file_name:
    #===============================================================================
    # Donor data
        (training_set, training_targets, test_set, test_targets) = get_preprocessed_data(file_name)
     
        # Setup up the parameters
        n_input_neurons = len(training_set[0])
        n_hidden_neurons = 40
        n_output_neurons = 1
        learning_rate = 0.9
        n_repeat_times = 5
     
        train_n_hidden_neurons = [40, 50, 60, 70, 80, 90]
        #=======================================================================
        # for  n_hidden_neurons in train_n_hidden_neurons:
        #     cv_measures =  np.repeat(0, 6, 0)  
        #     valid_measures =  np.repeat(0, 6, 0) 
        #     start = time.time()
        #     for i in range(0, n_repeat_times):
        #         neural_network_donor = NeuralNetwork(n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate)                
        #         #neural_network_donor.train_epoch(training_set, training_targets, 2)
        #         neural_network_donor.train_epoch_with_cross_validation(training_set, training_targets, 20)
        #         
        #         cv_measures = np.vstack((cv_measures, [neural_network_donor.mse_cv, neural_network_donor.error_cv, neural_network_donor.accuracy_cv, neural_network_donor.recall_cv, neural_network_donor.precision_cv, neural_network_donor.specificity_cv]))
        #         valid_measures = np.vstack((valid_measures, [neural_network_donor.mse_valid, neural_network_donor.error_valid, neural_network_donor.accuracy_valid, neural_network_donor.recall_valid, neural_network_donor.precision_valid, neural_network_donor.specificity_valid]))
        #     
        #     end = time.time()
        #     cv_measures = np.delete(cv_measures, 0, axis = 0)
        #     valid_measures = np.delete(valid_measures, 0, axis = 0)
        #     
        #     print ("Number of hidden neurons: ", n_hidden_neurons)
        #     print("Training data: ")
        #     print("Min: ", np.around([min(x) for x in np.transpose(cv_measures)], 5))
        #     print("Mean: ", np.around([np.mean(x) for x in np.transpose(cv_measures)], 5))
        #     print("Median: ", np.around([np.median(x) for x in np.transpose(cv_measures)], 5))
        #     print("Max: ", np.around([max(x) for x in np.transpose(cv_measures)], 5))
        #     
        #     print("Validation data: ")
        #     print("Min: ", np.around([min(x) for x in np.transpose(valid_measures)], 5))
        #     print("Mean: ", np.around([np.mean(x) for x in np.transpose(valid_measures)], 5))
        #     print("Median: ", np.around([np.median(x) for x in np.transpose(valid_measures)], 5))
        #     print("Max: ", np.around([max(x) for x in np.transpose(valid_measures)], 5))
        #     
        #     print ("Time: ", end - start)
        #=======================================================================
        neural_network_donor = NeuralNetwork(n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate)
        start = time.time()
        neural_network_donor.train_epoch(training_set, training_targets, 1000)
        end = time.time()
        pred_train_donor = (neural_network_donor.mse, neural_network_donor.error, neural_network_donor.accuracy,
                        neural_network_donor.recall, neural_network_donor.precision, neural_network_donor.specificity)     
       
        pred_test_donor = neural_network_donor.predict(test_set, test_targets)
    
        print "Donor data:"
        print "MSE | Error | Accuracy | Recall | Precision | Specificity"
        print "Training prediction: ", pred_train_donor, " N of hidden neurons: ", n_hidden_neurons, " Time: ", (end - start)
        print "Test prediction: ", pred_test_donor 
    
        neural_network_donor.save_weights("input_hidden_w_donor.csv", "hidden_output_w_donor.csv")
        
        sys.exit()
    if "Akceptor" in file_name:
    #===============================================================================
    # Acceptor data
        (training_set, training_targets, test_set, test_targets) = get_preprocessed_data(file_name)
     
        # Setup up the parameters
        n_input_neurons = len(training_set[0])
        n_hidden_neurons = 90
        n_output_neurons = 1
        learning_rate = 0.9
        n_repeat_times = 20
     
        #=======================================================================
        # train_n_hidden_neurons = [70, 80, 90, 100, 110, 120, 130, 140]
        # for  n_hidden_neurons in train_n_hidden_neurons:
        #     cv_measures =  np.repeat(0, 6, 0)  
        #     valid_measures =  np.repeat(0, 6, 0) 
        #     for i in range(0, n_repeat_times):
        #         neural_network_acceptor = NeuralNetwork(n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate)
        #         start = time.time()
        #         #neural_network_acceptor.train_epoch(training_set, training_targets, 2)
        #         neural_network_acceptor.train_epoch_with_cross_validation(training_set, training_targets, 20)
        #         end = time.time()
        # 
        #         cv_measures = np.vstack((cv_measures, [neural_network_acceptor.mse_cv, neural_network_acceptor.error_cv, neural_network_acceptor.accuracy_cv, neural_network_acceptor.recall_cv, neural_network_acceptor.precision_cv, neural_network_acceptor.specificity_cv]))
        #         valid_measures = np.vstack((valid_measures, [neural_network_acceptor.mse_valid, neural_network_acceptor.error_valid, neural_network_acceptor.accuracy_valid, neural_network_acceptor.recall_valid, neural_network_acceptor.precision_valid, neural_network_acceptor.specificity_valid]))
        # 
        #     cv_measures = np.delete(cv_measures, 0, axis = 0)
        #     valid_measures = np.delete(valid_measures, 0, axis = 0)
        #     
        #     print ("Number of hidden neurons: ", n_hidden_neurons)
        #     print("Training data: ")
        #     print("Min: ", np.around([min(x) for x in np.transpose(cv_measures)], 2))
        #     print("Mean: ", np.around([np.mean(x) for x in np.transpose(cv_measures)], 2))
        #     print("Median: ", np.around([np.median(x) for x in np.transpose(cv_measures)], 2))
        #     print("Max: ", np.around([max(x) for x in np.transpose(cv_measures)], 2))
        #     
        #     print("Validation data: ")
        #     print("Min: ", np.around([min(x) for x in np.transpose(valid_measures)], 2))
        #     print("Mean: ", np.around([np.mean(x) for x in np.transpose(valid_measures)], 2))
        #     print("Median: ", np.around([np.median(x) for x in np.transpose(valid_measures)], 2))
        #     print("Max: ", np.around([max(x) for x in np.transpose(valid_measures)], 2))
        #=======================================================================
            
        neural_network_acceptor = NeuralNetwork(n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate)
        start = time.time()
        neural_network_acceptor.train_epoch(training_set, training_targets, 1000)
        end = time.time()
        pred_train_acceptor = (neural_network_acceptor.mse, neural_network_acceptor.error, neural_network_acceptor.accuracy,
                           neural_network_acceptor.recall, neural_network_acceptor.precision, neural_network_acceptor.specificity)
    
        pred_test_acceptor = neural_network_acceptor.predict(test_set, test_targets)
    
        print "Acceptor data:"  
        print "MSE | Error | Accuracy | Recall | Precision | Specificity" 
        print "Training prediction: ", pred_train_acceptor, " N of hidden neurons: ", n_hidden_neurons, " Time: ", (end - start)
        print "Test prediction: ", pred_test_acceptor  
    
        neural_network_acceptor.save_weights("input_hidden_w_acceptor.csv", "hidden_output_w_acceptor.csv") 
        
        sys.exit()
    else:
        print "Please provide the file name with one of the word 'Donor' or Acceptor' to let the program choose the proper NN"
