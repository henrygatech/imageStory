import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
                # Initializing and empty list for predictions
        predicted_sentence = []
        
        # iterating max_len times
        for index in range(max_len):
            
            # Running through the LSTM layer
            lstm_out, states = self.lstm(inputs, states)

            # Running through the linear layer
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            
            # Getting the maximum probabilities
            target = outputs.max(1)[1]
            
            # Appending the result into a list
            predicted_sentence.append(target.item())
            
            # Updating the input
            inputs = self.embed(target).unsqueeze(1)
            
        return predicted_sentence
        
        '''
        ret = []
        for i in range(max_len):
            print(i)
            hiddens, _ = self.lstm(inputs)
            print(hiddens.shape)
            hiddens = hiddens.squeeze(1)
            print(hiddens.shape)
            outputs = self.linear(hiddens)
            print(outputs.shape)
            value, indice = outputs.max(1)
            print(indice)
            ret.append(indice.item())
            inputs = self.embed(indice).unsqueeze(1)
            print('embedded', inputs.shape)
        return ret
        
        '''
        