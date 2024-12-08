import torch
import torch.nn as nn
import random


## Encoder Model for Input Text, will remain same for Attention and Non-Attention Models
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, embeddings):
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze = True, padding_idx = 0)
        self.lstm_layer = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, bias = True, bidirectional = True, batch_first = True)
        
    def forward(self, X):
        embedded_input = self.embedding_layer(X)
        output, (hidden, cell) = self.lstm_layer(embedded_input)
        return output, hidden, cell
    

## Decoder Model without Attention to produce output from the hidden context
class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, embeddings):
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze = True, padding_idx = 0)
        self.lstm_layer = nn.LSTM(input_size = input_dim, hidden_size = 2 * hidden_dim, bias = True, bidirectional = False, batch_first = True)
        self.fc_layer = nn.Linear(in_features = 2 * hidden_dim, out_features = output_dim)
    
    def forward(self, X, hidden, cell):
        embedded_input = self.embedding_layer(X)
        output, (hidden, cell) = self.lstm_layer(embedded_input, (hidden, cell))
        prediction = self.fc_layer(output)
        return prediction, hidden, cell
    

### Model Class to pack the encoder and decoder
class Summarizer(nn.Module):
    
    def __init__(self, encoder, decoder, _SOS_token, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = 0.6
        self.device = device
        self.SOS_token = _SOS_token
    
    
    def forward(self, X, y):
        
        batch_size = X.shape[0]
        _, hidden, cell = self.encoder.forward(X)
        dec_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_outputs = []
        
        hidden = torch.cat((hidden[0], hidden[1]), dim = 1).unsqueeze(0)
        cell = torch.cat((cell[0], cell[1]), dim = 1).unsqueeze(0)
        
        for t in range(0, y.shape[1]):
            decoder_output, hidden, cell = self.decoder.forward(dec_input, hidden, cell)
            use_teacher = random.random() < self.teacher_forcing_ratio
            _, best_output = decoder_output.topk(1)
            dec_input = y[:, t].unsqueeze(1).detach() if use_teacher else best_output.squeeze(-1).detach()
            decoder_outputs.append(decoder_output)
            
        decoder_outputs = torch.cat(decoder_outputs, dim=1).to(self.device)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
            
        return decoder_outputs
    

    def predict(self, X, maxlen):

        batch_size = X.shape[0]
        _, hidden, cell = self.encoder.forward(X)
        dec_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_outputs = []
        
        hidden = torch.cat((hidden[0], hidden[1]), dim = 1).unsqueeze(0)
        cell = torch.cat((cell[0], cell[1]), dim = 1).unsqueeze(0)
        
        for t in range(0, maxlen):
            decoder_output, hidden, cell = self.decoder.forward(dec_input, hidden, cell)
            _, best_output = decoder_output.topk(1)
            dec_input = best_output.squeeze(-1).detach()
            decoder_outputs.append(decoder_output)
            
        decoder_outputs = torch.cat(decoder_outputs, dim=1).to(self.device)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
            
        return decoder_outputs


### Attention Mechanism
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = nn.functional.softmax(scores, dim=-1)
        
        context = torch.bmm(weights, keys)

        return context, weights
    

### Decoder Model with Attention produces output using Hidden Context + Key-Query Vectors 
class AttnDecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embeddings):
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze = True, padding_idx = 0)
        self.attention_layer = BahdanauAttention(2 * hidden_dim)
        self.lstm_layer = nn.LSTM(input_size = input_dim + 2 * hidden_dim, hidden_size = 2 * hidden_dim, bias = True, batch_first = True)
        self.fc_layer = nn.Linear(in_features = 2 * hidden_dim, out_features = output_dim)
        
    def forward(self, X, decoder_hidden, decoder_cell, encoder_outputs):
        
        embedded_input = self.embedding_layer(X)
        query = decoder_hidden.permute(1, 0, 2)
        hidden_context, hidden_attn_weights = self.attention_layer(query, encoder_outputs)
        input_lstm = torch.cat((embedded_input, hidden_context), dim = 2)
        output, (decoder_hidden, decoder_cell) = self.lstm_layer(input_lstm, (decoder_hidden, decoder_cell))
        prediction = self.fc_layer(output)
        
        return prediction, decoder_hidden, decoder_cell, hidden_attn_weights


### Model class to pack Encoder and Decoder
class AttnSummarizer(nn.Module):

    def __init__(self, encoder, decoder, _SOS_token, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = 0.6
        self.device = device
        self.SOS_token = _SOS_token
    
    def forward(self, X, y):
        
        batch_size = y.shape[0]
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(X)
        dec_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_outputs = []
        
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim = 1).unsqueeze(0)
        decoder_cell = torch.cat((encoder_cell[0], encoder_cell[1]), dim = 1).unsqueeze(0)
        
        for t in range(0, y.shape[1]):
            decoder_output, decoder_hidden, decoder_cell, attn_weights = self.decoder.forward(dec_input, decoder_hidden, decoder_cell, encoder_outputs)
            use_teacher = random.random() < self.teacher_forcing_ratio
            _, best_output = decoder_output.topk(1)
            dec_input = y[:, t].unsqueeze(1).detach() if use_teacher else best_output.squeeze(-1).detach()
            decoder_outputs.append(decoder_output)
            
        decoder_outputs = torch.cat(decoder_outputs, dim=1).to(self.device)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
            
        return decoder_outputs

    def predict(self, X, maxlen):
        
        batch_size = X.shape[0]
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder.forward(X)
        dec_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.SOS_token)
        decoder_outputs = []
        
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim = 1).unsqueeze(0)
        decoder_cell = torch.cat((encoder_cell[0], encoder_cell[1]), dim = 1).unsqueeze(0)
        
        for t in range(0, maxlen):
            decoder_output, decoder_hidden, decoder_cell, attn_weights = self.decoder.forward(dec_input, decoder_hidden, decoder_cell, encoder_outputs)
            _, best_output = decoder_output.topk(1)
            dec_input = best_output.squeeze(-1).detach()
            decoder_outputs.append(decoder_output)
            
        decoder_outputs = torch.cat(decoder_outputs, dim=1).to(self.device)
        decoder_outputs = nn.functional.log_softmax(decoder_outputs, dim=-1)
            
        return decoder_outputs