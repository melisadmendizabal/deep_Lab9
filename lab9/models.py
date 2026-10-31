"""Redes propias en PyTorch. Todas comparten la firma forward(x, x_fut, y=None) -> (B, H).

x     : (B, L, 1 + n_cal)  canal 0 = y normalizada, resto = calendario de las horas pasadas
x_fut : (B, H, n_cal)      calendario de las horas a pronosticar
y     : (B, H)             solo lo usan los modelos con teacher forcing durante el entrenamiento
"""
import torch
from torch import nn


class MLP(nn.Module):
    """MLP multi-salida: aplana la ventana (y opcionalmente el calendario futuro) y devuelve H pasos."""

    def __init__(self, L, n_feat, n_cal, H, hidden=512, n_layers=2, dropout=0.1, use_cov=True):
        super().__init__()
        self.use_cov = use_cov
        d_in = L * n_feat + H * n_cal if use_cov else L
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d_in = hidden
        layers.append(nn.Linear(d_in, H))
        self.net = nn.Sequential(*layers)

    def forward(self, x, x_fut, y=None):
        z = torch.cat([x.flatten(1), x_fut.flatten(1)], 1) if self.use_cov else x[..., 0]
        return self.net(z)


class MovingAvg(nn.Module):
    """Media móvil con relleno replicado en los extremos (tendencia de DLinear)."""

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1)

    def forward(self, z):                                   # z: (B, L)
        front = z[:, :1].repeat(1, (self.kernel - 1) // 2)
        back = z[:, -1:].repeat(1, self.kernel // 2)
        return self.pool(torch.cat([front, z, back], 1)[:, None])[:, 0]


class DLinear(nn.Module):
    """DLinear (Zeng et al., 2023): serie = tendencia (media móvil) + resto; una capa lineal
    L -> H para cada componente, compartida por todas las series. Solo usa la serie objetivo."""

    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.trend = nn.Linear(L, H)
        self.seasonal = nn.Linear(L, H)

    def forward(self, x, x_fut=None, y=None):
        z = x[..., 0]
        trend = self.decomp(z)
        return self.trend(trend) + self.seasonal(z - trend)


class NLinear(nn.Module):
    """NLinear: resta el último valor observado, aplica una capa lineal L -> H y lo vuelve a sumar.
    Esa resta es una normalización por ventana mínima que corrige cambios de nivel."""

    def __init__(self, L, H):
        super().__init__()
        self.linear = nn.Linear(L, H)

    def forward(self, x, x_fut=None, y=None):
        z = x[..., 0]
        last = z[:, -1:]
        return self.linear(z - last) + last


def _rnn(cell, n_in, hidden, n_layers, dropout):
    rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
    return rnn(n_in, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)


class RNNMultiOutput(nn.Module):
    """Estrategia multi-salida: LSTM/GRU lee la ventana y una capa densa devuelve los H pasos.
    Con use_cov=True la capa densa también recibe el calendario de las H horas futuras."""

    def __init__(self, n_feat, n_cal, H, hidden=128, n_layers=2, dropout=0.1, cell="lstm", use_cov=True):
        super().__init__()
        self.use_cov = use_cov
        self.rnn = _rnn(cell, n_feat, hidden, n_layers, dropout)
        self.head = nn.Linear(hidden + (H * n_cal if use_cov else 0), H)

    def forward(self, x, x_fut, y=None):
        out, _ = self.rnn(x)                                # out: (B, L, hidden)
        z = out[:, -1]
        if self.use_cov:
            z = torch.cat([z, x_fut.flatten(1)], 1)
        return self.head(z)


class RNNAutoregressive(nn.Module):
    """Estrategia autorregresiva: el modelo predice solo t+1.

    Entrenamiento con teacher forcing: la secuencia de entrada se extiende con los valores
    reales del horizonte (desplazados un paso), así cada salida ve la verdad del paso anterior.
    Inferencia: se predice un paso, la predicción se realimenta como entrada y se repite H veces;
    los errores de cada paso se acumulan en los siguientes.
    """

    def __init__(self, n_feat, n_cal, H, hidden=128, n_layers=2, dropout=0.1, cell="lstm"):
        super().__init__()
        self.H = H
        self.rnn = _rnn(cell, n_feat, hidden, n_layers, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, x_fut, y=None):
        L = x.shape[1]
        if self.training and y is not None:
            # Entrada en el paso L+k: [y_real(L+k), calendario(L+k)], k = 0..H-2
            fut = torch.cat([y[:, :-1, None], x_fut[:, :-1]], -1)
            out, _ = self.rnn(torch.cat([x, fut], 1))
            return self.head(out[:, L - 1:]).squeeze(-1)    # (B, H)
        out, state = self.rnn(x)
        pred = self.head(out[:, -1])                         # (B, 1)
        preds = [pred]
        for k in range(self.H - 1):
            inp = torch.cat([pred, x_fut[:, k]], -1)[:, None]   # realimenta su propia predicción
            out, state = self.rnn(inp, state)
            pred = self.head(out[:, -1])
            preds.append(pred)
        return torch.cat(preds, 1)


class Seq2Seq(nn.Module):
    """Encoder-decoder: el encoder resume la ventana en su estado (h, c); el decoder recorre las
    H horas futuras recibiendo su calendario. Con feed_prev=True el decoder además recibe el valor
    del paso anterior (real con probabilidad `teacher_forcing` al entrenar, predicho al inferir).
    """

    def __init__(self, n_feat, n_cal, H, hidden=128, n_layers=2, dropout=0.1, cell="lstm",
                 feed_prev=False, teacher_forcing=0.5):
        super().__init__()
        self.feed_prev, self.teacher_forcing = feed_prev, teacher_forcing
        self.encoder = _rnn(cell, n_feat, hidden, n_layers, dropout)
        self.decoder = _rnn(cell, n_cal + int(feed_prev), hidden, n_layers, dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, x_fut, y=None):
        _, state = self.encoder(x)
        if not self.feed_prev:
            out, _ = self.decoder(x_fut, state)             # las H horas en paralelo
            return self.head(out).squeeze(-1)
        prev = x[:, -1, :1]                                  # último valor observado
        preds = []
        for k in range(x_fut.shape[1]):
            out, state = self.decoder(torch.cat([prev, x_fut[:, k]], -1)[:, None], state)
            pred = self.head(out[:, -1])
            preds.append(pred)
            use_truth = self.training and y is not None and torch.rand(()) < self.teacher_forcing
            prev = y[:, k:k + 1] if use_truth else pred
        return torch.cat(preds, 1)


# Constructores con firma común (L, n_feat, n_cal, H, **hiperparámetros). Se usan con
# functools.partial para que TorchForecaster registre los hiperparámetros de arquitectura.
def build_mlp(L, n_feat, n_cal, H, **kw):
    return MLP(L, n_feat, n_cal, H, **kw)


def build_dlinear(L, n_feat, n_cal, H, **kw):
    return DLinear(L, H, **kw)


def build_nlinear(L, n_feat, n_cal, H, **kw):
    return NLinear(L, H, **kw)


def build_rnn_mo(L, n_feat, n_cal, H, **kw):
    return RNNMultiOutput(n_feat, n_cal, H, **kw)


def build_rnn_ar(L, n_feat, n_cal, H, **kw):
    return RNNAutoregressive(n_feat, n_cal, H, **kw)


def build_seq2seq(L, n_feat, n_cal, H, **kw):
    return Seq2Seq(n_feat, n_cal, H, **kw)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
