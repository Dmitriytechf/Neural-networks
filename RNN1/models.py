import torch
import torch.nn as nn


class RecurrentModel(nn.Module):
    """
    Рекуррентная нейронная сеть для прогнозирования временных рядов.
    Поддержка двух архитектур:
        -LSTM - Запоминает долгосрочные зависимости
        -GRU - Более легковесная альтернатива LSTM
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, 
                 dropout=0.2, rnn_type='LSTM'):
        super(RecurrentModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers # Количество RNN слоев
        self.rnn_type = rnn_type # Тип рекуррентного слоя

        # Выбор типа RNN слоя
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout)
        else:
            raise ValueError('Поддерживаются только LSTM и GRU')

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """
        Прямой проход через RNN сеть
        """
        # RNN проход
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Берем только последний временной шаг для прогнозирования
        last_output = rnn_out[:, -1, :]
        
        output = self.dropout(last_output) # Применяем dropout для регуляризации
        output = self.linear(output)
        
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Инициализация начального скрытого состояния
        """
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                     weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        else:
            hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        
        return hidden
