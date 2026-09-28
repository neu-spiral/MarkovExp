from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import functional_call, jacrev, vmap



class MLPMeasurement(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(32, 32),
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        if activation == "tanh":
            self._act = torch.tanh
        elif activation == "relu":
            self._act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        # Build MLP: [input_dim] -> hidden_dims -> 1
        dims = (self.input_dim,) + self.hidden_dims + (1,)
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)

    # -----------------------------
    # β utilities
    # -----------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flatten_beta(self) -> np.ndarray:
        """Return current model params as a flat numpy vector."""
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()])
    
    def _beta_to_params(self, beta_vec: torch.Tensor):
        """
        Create an OrderedDict {param_name: tensor_view} that is *connected*
        to beta_vec (no detach, no copy_, no no_grad).
        """
        beta_vec = beta_vec.reshape(-1)
        expected = self.num_params()
        if beta_vec.numel() != expected:
            raise ValueError(f"beta_vec has length {beta_vec.numel()} but expected {expected}")

        params = OrderedDict()
        offset = 0
        for name, p in self.named_parameters():
            n = p.numel()
            chunk = beta_vec[offset:offset+n].view_as(p).to(device=p.device, dtype=p.dtype)
            params[name] = chunk
            offset += n
        return params

    def load_beta(self, beta_vec) -> None:
        if isinstance(beta_vec, np.ndarray):
            beta = torch.from_numpy(beta_vec.reshape(-1))
        else:
            beta = beta_vec.reshape(-1).detach().cpu()

        expected = self.num_params()
        if beta.numel() != expected:
            raise ValueError(f"beta_vec has length {beta.numel()} but expected {expected}")

        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                chunk = beta[offset:offset + numel].to(p.device).type_as(p)
                p.copy_(chunk.view_as(p))
                offset += numel

    # -----------------------------
    # Forward h(X, β)
    # -----------------------------
    def forward(self, X: torch.Tensor, beta_vec=None) -> torch.Tensor:
        """
        X: (n, d_x)
        beta_vec: optional flat β. If provided, loads it first.
        Returns y: (n,)
        """
        if beta_vec is None:
            h = X
            L = len(self.layers)
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < L - 1:
                    h = self._act(h)
            return h.squeeze(-1)

        params = self._beta_to_params(beta_vec)
        buffers = dict(self.named_buffers())  # usually empty here

        y = functional_call(self, (params, buffers), (X,))  # uses self.forward's beta_vec=None branch
        return y.squeeze(-1)

    def jacobian_wrt_beta(self, X: torch.Tensor, beta_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute H = ∂h(X,β)/∂β at beta_vec.
        Returns H with shape (n, d_beta).

        Requires PyTorch 2.x torch.func. If unavailable, raises an error.
        """
        if functional_call is None or jacrev is None or vmap is None:
            raise RuntimeError("torch.func is required (PyTorch 2.x).")

        beta_vec = beta_vec.reshape(-1)

        def f(beta, x_single):
            y = self.forward(x_single.unsqueeze(0), beta)  # (1,)
            return y.squeeze(0)  # scalar

        H = vmap(jacrev(f), in_dims=(None, 0))(beta_vec, X)
        return H


class CNNMeasurement(nn.Module):

    def __init__(
        self,
        input_dim: int,
        channels=(4, 4),
        kernel_sizes=(3, 3),
        activation: str = "tanh",
        use_bias: bool = True,
        pool_out_len: int = 1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.use_bias = bool(use_bias)
        self.pool_out_len = int(pool_out_len)

        if len(self.channels) == 0:
            raise ValueError("channels must have at least one entry")
        if len(self.channels) != len(self.kernel_sizes):
            raise ValueError("channels and kernel_sizes must have same length")
        if self.pool_out_len < 1:
            raise ValueError("pool_out_len must be >= 1")

        if activation == "tanh":
            self._act = torch.tanh
        elif activation == "relu":
            self._act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        convs = []
        in_ch = 1
        for out_ch, k in zip(self.channels, self.kernel_sizes):
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=self.use_bias))
            in_ch = out_ch
        self.convs = nn.ModuleList(convs)

        self.pool = nn.AdaptiveAvgPool1d(self.pool_out_len)

        head_in = self.channels[-1] * self.pool_out_len
        self.head = nn.Linear(head_in, 1, bias=True)

    # -----------------------------
    # β utilities
    # -----------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flatten_beta(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()])

    def _beta_to_params(self, beta_vec: torch.Tensor):
        beta_vec = beta_vec.reshape(-1)
        expected = self.num_params()
        if beta_vec.numel() != expected:
            raise ValueError(f"beta_vec has length {beta_vec.numel()} but expected {expected}")

        params = OrderedDict()
        offset = 0
        for name, p in self.named_parameters():
            n = p.numel()
            chunk = beta_vec[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
            params[name] = chunk
            offset += n
        return params

    def load_beta(self, beta_vec) -> None:
        if isinstance(beta_vec, np.ndarray):
            beta = torch.from_numpy(beta_vec.reshape(-1))
        else:
            beta = beta_vec.reshape(-1).detach().cpu()

        expected = self.num_params()
        if beta.numel() != expected:
            raise ValueError(f"beta_vec has length {beta.numel()} but expected {expected}")

        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                chunk = beta[offset:offset + numel].to(p.device).type_as(p)
                p.copy_(chunk.view_as(p))
                offset += numel

    # -----------------------------
    # Input shaping
    # -----------------------------
    def _ensure_conv1d_input(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            return X.unsqueeze(1)  # (n,1,d)
        if X.dim() == 3:
            return X
        raise ValueError(f"X must have shape (n,d) or (n,C,L); got {tuple(X.shape)}")

    # -----------------------------
    # Forward h(X, β)
    # -----------------------------
    def forward(self, X: torch.Tensor, beta_vec=None) -> torch.Tensor:
        if beta_vec is None:
            z = self._ensure_conv1d_input(X)  # (n,C,L)
            Lnum = len(self.convs)
            for i, conv in enumerate(self.convs):
                z = conv(z)
                z = self._act(z)

            z = self.pool(z)                     # (n, C_last, pool_out_len)
            z = z.flatten(start_dim=1)           # (n, C_last*pool_out_len)
            y = self.head(z)                     # (n,1)
            return y.squeeze(-1)

        params = self._beta_to_params(beta_vec)
        buffers = dict(self.named_buffers())
        y = functional_call(self, (params, buffers), (X,))
        return y.squeeze(-1)

    # -----------------------------
    # Jacobian wrt β
    # -----------------------------
    def jacobian_wrt_beta(self, X: torch.Tensor, beta_vec: torch.Tensor) -> torch.Tensor:
        if functional_call is None or jacrev is None or vmap is None:
            raise RuntimeError("torch.func is required (PyTorch 2.x).")

        beta_vec = beta_vec.reshape(-1)

        def f(beta, x_single):
            # x_single: (d,) or (C,L)
            if x_single.dim() == 1:
                x_in = x_single.unsqueeze(0)     # (1,d)
            elif x_single.dim() == 2:
                x_in = x_single.unsqueeze(0)     # (1,C,L)
            else:
                raise ValueError(f"Per-sample x must be (d,) or (C,L); got {tuple(x_single.shape)}")

            y = self.forward(x_in, beta)         # (1,)
            return y.squeeze(0)                  # scalar

        H = vmap(jacrev(f), in_dims=(None, 0))(beta_vec, X)
        return H


class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='tanh', use_bias=True):
        super().__init__()
        padding = kernel_size // 2  # same padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, bias=use_bias)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, bias=use_bias)
        # Shortcut projection if dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        if activation == 'tanh':
            self._act = torch.tanh
        elif activation == 'relu':
            self._act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

    def forward(self, x):
        # x: (batch, channels, length)
        identity = self.shortcut(x)
        out = self._act(self.conv1(x))
        out = self.conv2(out)
        out = self._act(out + identity)
        return out


class ResNetMeasurement(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels=(4, 4),
        kernel_sizes=(3, 3),
        activation: str = 'tanh',
        use_bias: bool = True,
        pool_out_len: int = 1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.pool_out_len = int(pool_out_len)

        assert len(self.channels) == len(self.kernel_sizes), \
            "channels and kernel_sizes must have the same length"

        if activation == 'tanh':
            self._act = torch.tanh
        elif activation == 'relu':
            self._act = F.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        init_padding = self.kernel_sizes[0] // 2
        self.init_conv = nn.Conv1d(
            1, self.channels[0], self.kernel_sizes[0],
            padding=init_padding, bias=use_bias
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.channels)):
            in_ch = self.channels[i - 1] if i > 0 else self.channels[0]
            out_ch = self.channels[i]
            ks = self.kernel_sizes[i]
            self.res_blocks.append(
                ResBlock1d(in_ch, out_ch, ks, activation=activation, use_bias=use_bias)
            )

        # Adaptive pooling + linear head
        self.pool = nn.AdaptiveAvgPool1d(self.pool_out_len)
        head_in = self.channels[-1] * self.pool_out_len
        self.head = nn.Linear(head_in, 1)

    # ---------------------------------------------------------
    # Beta utilities
    # ---------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flatten_beta(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in self.parameters()
        ])

    def _beta_to_params(self, beta_vec: torch.Tensor):
        beta_vec = beta_vec.reshape(-1)
        expected = self.num_params()
        if beta_vec.numel() != expected:
            raise ValueError(
                f"beta_vec has length {beta_vec.numel()} but expected {expected}"
            )
        params = OrderedDict()
        offset = 0
        for name, p in self.named_parameters():
            n = p.numel()
            chunk = beta_vec[offset:offset + n].view_as(p).to(
                device=p.device, dtype=p.dtype
            )
            params[name] = chunk
            offset += n
        return params

    def load_beta(self, beta_vec) -> None:
        if isinstance(beta_vec, np.ndarray):
            beta = torch.from_numpy(beta_vec.reshape(-1))
        else:
            beta = beta_vec.reshape(-1).detach().cpu()
        expected = self.num_params()
        if beta.numel() != expected:
            raise ValueError(
                f"beta_vec has length {beta.numel()} but expected {expected}"
            )
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                chunk = beta[offset:offset + numel].to(p.device).type_as(p)
                p.copy_(chunk.view_as(p))
                offset += numel

    # ---------------------------------------------------------
    # Forward h(X, beta)
    # ---------------------------------------------------------
    def forward(self, X: torch.Tensor, beta_vec=None) -> torch.Tensor:
        if beta_vec is not None:
            params = self._beta_to_params(beta_vec)
            buffers = dict(self.named_buffers())
            y = functional_call(self, (params, buffers), (X,))
            return y.squeeze(-1)

        h = X.unsqueeze(1)

        h = self._act(self.init_conv(h))

        for block in self.res_blocks:
            h = block(h)

        # Pool + flatten + head
        h = self.pool(h)              # (n, channels[-1], pool_out_len)
        h = h.view(h.size(0), -1)     # (n, channels[-1] * pool_out_len)
        h = self.head(h)              # (n, 1)
        return h.squeeze(-1)          # (n,)

    # ---------------------------------------------------------
    # Jacobian wrt beta
    # ---------------------------------------------------------
    def jacobian_wrt_beta(self, X: torch.Tensor, beta_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute H = dh(X, beta)/d(beta) at beta_vec.
        Returns H with shape (n, d_beta).
        """
        if functional_call is None or jacrev is None or vmap is None:
            raise RuntimeError("torch.func is required (PyTorch 2.x).")

        beta_vec = beta_vec.reshape(-1)

        def f(beta, x_single):
            y = self.forward(x_single.unsqueeze(0), beta)
            return y.squeeze(0)

        H = vmap(jacrev(f), in_dims=(None, 0))(beta_vec, X)
        return H
