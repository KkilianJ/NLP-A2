"""Frozen token embedding lookup and unidirectional final-valid-state LSTM."""
import torch
from torch import nn


class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, head="linear", head_hidden_size=64):
        super().__init__()
        if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] != 300:
            raise ValueError("Embedding matrix must have shape (vocabulary, 300).")
        if not torch.equal(embedding_matrix[0], torch.zeros_like(embedding_matrix[0])):
            raise ValueError("Padding row must contain only zeros.")
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=False, dropout=0.0)
        if head == "linear":
            # Keep the original module and state-dict keys for old checkpoints.
            self.classifier = nn.Linear(hidden_size, 1)
        elif head == "mlp":
            if head_hidden_size != 64:
                raise ValueError("The controlled MLP head must have exactly 64 hidden units.")
            self.classifier = nn.Sequential(nn.Linear(hidden_size, head_hidden_size),
                                            nn.ReLU(), nn.Linear(head_hidden_size, 1))
        else:
            raise ValueError(f"Unknown classification head: {head}")
        self.head = head
        self.head_hidden_size = head_hidden_size if head == "mlp" else None

    def forward(self, token_ids, lengths):
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("Expected token IDs with shape (batch, at least 1 position).")
        if lengths.ndim != 1 or lengths.shape[0] != token_ids.shape[0]:
            raise ValueError("Expected one true sequence length per review.")
        if torch.any(lengths < 0) or torch.any(lengths > token_ids.shape[1]):
            raise ValueError("A sequence length is outside the padded batch width.")
        # In a forward-only LSTM, an output at time t cannot depend on later
        # right-padding. Gather the last valid time step, not h_n after padding.
        sequence_output, _ = self.lstm(self.embedding(token_ids))
        lengths_on_device = lengths.to(device=sequence_output.device, dtype=torch.long)
        last_index = (lengths_on_device - 1).clamp(min=0)
        row = torch.arange(token_ids.shape[0], device=sequence_output.device)
        final = sequence_output[row, last_index]
        # Empty reviews use a zero hidden state; their logit comes from the
        # classifier applied to that state, not from processing a padding token.
        final = final * (lengths_on_device > 0).unsqueeze(1).to(final.dtype)
        return self.classifier(final).squeeze(-1)


def model_from_checkpoint(checkpoint):
    """Reconstruct old linear and new head-aware checkpoints without training."""
    head = checkpoint.get("head", "linear")
    head_hidden_size = checkpoint.get("head_hidden_size") or 64
    model = SentimentLSTM(checkpoint["embedding_table"], checkpoint["hidden_size"],
                          head=head, head_hidden_size=head_hidden_size)
    status = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if set(status.missing_keys) != {"embedding.weight"} or status.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {status}")
    return model
