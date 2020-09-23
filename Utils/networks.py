import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from .utils import *

# The style transfer network
class StyleTransferNetwork(nn.Module):
  def __init__(self,
               device, # "cpu" for cpu, "cuda" for gpu
               enc_state_dict, # The state dict of the pretrained vgg19
               learning_rate=1e-4,
               learning_rate_decay=5e-5, # Decay parameter for the learning rate
               gamma=2.0, # Controls importance of StyleLoss vs ContentLoss, Loss = gamma*StyleLoss + ContentLoss
               train=True, # Wether or not network is training
               load_fromstate=False, # Load from checkpoint?
               load_path=None # Path to load checkpoint
               ):
    super().__init__()

    assert device in ["cpu", "cuda"]
    if load_fromstate and not os.path.isfile(load_path):
      raise ValueError("Checkpoint file not found")


    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.train = train
    self.gamma = gamma

    self.encoder = Encoder(enc_state_dict, device) # A pretrained vgg19 is used as the encoder
    self.decoder = Decoder().to(device)

    self.optimiser = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
    self.iters = 0

    if load_fromstate:
      state = torch.load(load_path)
      self.decoder.load_state_dict(state["Decoder"])
      self.optimiser.load_state_dict(state["Optimiser"])
      self.iters = state["iters"]


  def set_train(self, boolean): # Change state of network
    assert type(boolean) == bool
    self.train = boolean

  def adjust_learning_rate(self, optimiser, iters): # Simple learning rate decay
    lr = learning_rate / (1.0 + learning_rate_decay * iters)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr

  def forward(self, style, content, alpha=1.0): # Alpha can be used while testing to control the importance of the transferred style

    # Encode style and content
    layers_style = self.encoder(style, self.train) # if train: returns all states
    layer_content = self.encoder(content, False) # for the content only the last layer is important

    # Transfer Style
    if self.train:
      style_applied = AdaIn(layer_content, layers_style[-1]) # Last layer is "style" layer
    else:
      style_applied = alpha*AdaIn(layer_content, layers_style) + (1-alpha)*layer_content # Alpha controls magnitude of style

    # Scale up
    style_applied_upscaled = self.decoder(style_applied)
    if not self.train:
      return style_applied_upscaled # When not training return transformed image

    # Compute Loss
    layers_style_applied = self.encoder(style_applied_upscaled, self.train)

    content_loss = Content_loss(layers_style_applied[-1], layer_content)
    style_loss = Style_loss(layers_style_applied, layers_style)

    loss_comb = content_loss + self.gamma*style_loss

    return loss_comb, content_loss, style_loss

# The decoder is a reversed vgg19 up to ReLU 4.1. To note is that the last layer is not activated.

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.padding = nn.ReflectionPad2d(padding=1) # Using reflection padding as described in vgg19
    self.UpSample = nn.Upsample(scale_factor=2, mode="nearest")

    self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

    self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

    self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)

    self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0)


  def forward(self, x):
    out = self.UpSample(F.relu(self.conv4_1(self.padding(x))))

    out = F.relu(self.conv3_1(self.padding(out)))
    out = F.relu(self.conv3_2(self.padding(out)))
    out = F.relu(self.conv3_3(self.padding(out)))
    out = self.UpSample(F.relu(self.conv3_4(self.padding(out))))

    out = F.relu(self.conv2_1(self.padding(out)))
    out = self.UpSample(F.relu(self.conv2_2(self.padding(out))))

    out = F.relu(self.conv1_1(self.padding(out)))
    out = self.conv1_2(self.padding(out))
    return out

# A vgg19 Sequential which is used up to Relu 4.1. To note is that the
# first layer is a 3,3 convolution, different from a standard vgg19

class Encoder(nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        self.vgg19 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True), # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True), # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1), # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True), # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True)
            ).to(device)

        self.vgg19.load_state_dict(state_dict)

        encoder_children = list(self.vgg19.children())
        self.EncoderList = nn.ModuleList([nn.Sequential(*encoder_children[:4]), # Up to Relu 1.1
                                          nn.Sequential(*encoder_children[4:11]), # Up to Relu 2.1
                                          nn.Sequential(*encoder_children[11:18]), # Up to Relu 3.1
                                          nn.Sequential(*encoder_children[18:31]), # Up to Relu 4.1, also the
                                          ])                                       # input for the decoder

    def forward(self, x, intermediates=False): # if training use intermediates = True, to get the output of
        states = []                            # all the encoder layers to calculate the style loss
        for i in range(len(self.EncoderList)):
            x = self.EncoderList[i](x)

            if intermediates:       # All intermediate states get saved in states
                states.append(x)
        if intermediates:
            return states
        return x
