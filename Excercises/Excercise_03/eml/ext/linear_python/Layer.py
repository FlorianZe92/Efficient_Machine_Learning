import torch

class Layer( torch.nn.Module):
    def __init__(self, i_n_features_input, i_n_features_output):
        super( Layer, self).__init__()

        self.m_weights = torch.nn.Parameter (torch.Tensor(i_n_features_input, i_n_features_output))

        print(self.m_weights.size())

        print("called __init__")

    def forward(self, i_input):
        l_res = torch.matmul(i_input, self.m_weights)
        return l_res

