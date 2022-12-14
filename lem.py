import torch
import torch.nn as nn
import math

class LEMCell(nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMCell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.dt = dt
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)

        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)

        z = (1.-ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1.-ms_dt_bar)* y + ms_dt_bar * torch.tanh(self.transform_z(z)+i_z)

        return y, z

class LEM(nn.Module):
    def __init__(self, ninp, nhid, nout, dt=1.):
        super(LEM, self).__init__()
        self.nhid = nhid
        self.cell = LEMCell(ninp,nhid,dt)
        

    def forward(self, input):
        ## initialize hidden states
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        for x in input:
            y, z = self.cell(x,y,z)
        return y
    
class SeqLEM(nn.Module):
    def __init__(self, ninp, nhid, dt=1.):
        super(SeqLEM, self).__init__()
        self.nhid = nhid
        self.cell = LEMCell(ninp,nhid,dt)
        

    def forward(self, input):
      y_ = list()
      z_ = list()
      ## initialize hidden states
      y = input.data.new(input.size(1), self.nhid).zero_()
      z = input.data.new(input.size(1), self.nhid).zero_()

      for x in input:
          y, z = self.cell(x,y,z)
          y_.append(y)
          z_.append(z)
      return torch.stack(y_)


class BidirectionalSeqLEM(nn.Module):
    def __init__(self, ninp, nhid, dt=1.):
        super(BidirectionalSeqLEM, self).__init__()
        self.nhid = nhid
        self.forward_ = LEMCell(ninp, nhid, dt)
        self.backward_ = LEMCell(ninp, nhid, dt)
        

    def forward(self, input):
      ys = list()
      zs = list()
      ## initialize hidden states
      y = input.data.new(input.size(1), self.nhid).zero_()
      z = input.data.new(input.size(1), self.nhid).zero_()

      y_ = input.data.new(input.size(1), self.nhid).zero_()
      z_ = input.data.new(input.size(1), self.nhid).zero_()

      max_len = input.shape[0]
      for i in range(max_len):
          y, z = self.forward_(input[i, :, :], y, z)
          y_, z_ = self.backward_(input[max_len - i - 1, :, :], y_, z_)
          ys.append(torch.cat([y, y_], dim=-1))
          zs.append(torch.cat([z, z_], dim=-1))
      return torch.stack(ys)
