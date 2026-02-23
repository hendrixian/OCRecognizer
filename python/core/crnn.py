import torch
import torch.nn as nn

class CRNN(nn.Module):
    """CRNN for OCR with CTC loss"""

    def __init__(self, img_height=32, num_classes=10, hidden_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.ReLU(inplace=True)
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)

    def forward(self, x):
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        if height != 1:
            raise ValueError(f'Expected height 1 after CNN, got {height}')
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)

        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        output = nn.functional.log_softmax(output, dim=2)
        return output
