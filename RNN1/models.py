import torch
import torch.nn as nn


# Пока не используем
class Attention(nn.Module):
    """
    Механизм внимания (attention mechanism)
    Вычисляет веса внимания для каждого временного 
    шага в последовательности. Позволяет нейронной сети 
    "фокусироваться" на наиболее важных частях входной 
    последовательности при принятии решения.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rnn_outputs):
        """
        Returns:
        --------
        context : тензор(torch.Tensor)
            Взвешенная сумма выходов RNN, форма: (batch_size, hidden_size)
        attention_weights : тензор(torch.Tensor)
            Веса внимания для каждого временного шага, форма: (batch_size, seq_len)
        """
        # Вычисляем важность для каждого временного шага
        attention_scores = self.attention(rnn_outputs)
        attention_scores = attention_scores.squeeze(-1)
        
        # Применяем softmax для получения вероятностного распределения
        attention_weights = self.softmax(attention_scores)
        
        # Вычисляем взвешенную сумму (контекстный вектор)
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


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

        self.hidden_size = hidden_size # Количество нейронов
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
        Прямой проход через RNN сеть.
        Обрабатывает входную последовательность, обновляет скрытое состояние
        и возвращает прогноз для следующего временного шага.
        
        Returns:
        --------
        output : torch.Tensor
            Прогноз для следующего временного шага после последовательности.
            Форма: (batch_size, 1)
        
        hidden : torch.Tensor или tuple of Tensors
            Обновленное скрытое состояние после обработки последовательности.
            Может быть передано для обработки следующей части последовательности.
            Форма совпадает с входным hidden.
        """
        # RNN проход - получаем все скрытые состояния
        rnn_out, hidden = self.rnn(x, hidden) 
        
        # Берем только последний временной шаг для прогнозирования
        last_output = rnn_out[:, -1, :]
        
        output = self.dropout(last_output) # Применяем dropout для регуляризации
        output = self.linear(output)
        
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Инициализация начального скрытого состояния.
        Скрытое состояние представляет собой "память" сети о предыдущих 
        временных шагах. Нулевая инициализация означает, что каждая новая
        последовательность обрабатывается с чистого состояния.
        
        Parameters:
        -----------
        batch_size : int
            Размер батча данных (количество параллельно обрабатываемых последовательностей)
        
        Каждая новая последовательность начинается с "чистой памяти". То есть, 
        сеть начинает обработку каждой новой последовательности без "предыдущего опыта".
        """
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                     weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        else:
            hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()

        return hidden
