


class MLP(ExpandableModel):

    def forward(self, x):

        # pass the data through all layers of the network
        for module in self.modulelist:
            x = module(x)

        self.y = x

        return self.y

    def build(self):

        self.modulelist = nn.ModuleList()

        self.modulelist.append(nn.Linear(self.input_size, self.hidden_size))
        self.modulelist.append(nn.ReLU())
        self.modulelist.append(nn.Linear(self.hidden_size, self.output_size))

    # initialize weights in the network in the same manner as in:
    # https://github.com/ariseff/overcoming-catastrophic/blob/afea2d3c9f926d4168cc51d56f1e9a92989d7af0/model.py#L7
    @staticmethod
    def init_weights(m):

        # This function is intended to mimic the behavior of TensorFlow's tf.truncated_normal(), returning
        # a tensor of the specified shape containing values sampled from a truncated normal distribution with the
        # specified mean and standard deviation. Sampled values which fall outside of the range of +/- 2 standard deviations
        # from the mean are dropped and re-picked.
        def trunc_normal_weights(shape, mean=0.0, stdev=0.1):

            num_samples = 1

            for dim in list(shape):
                num_samples *= dim

            a, b = ((mean - 2 * stdev) - mean) / stdev, ((mean + 2 * stdev) - mean) / stdev

            samples = stats.truncnorm.rvs(a, b, scale=stdev, loc=mean, size=num_samples)

            return torch.Tensor(samples.reshape(tuple(shape)))

        if type(m) == nn.Linear:
            m.weight.data.copy_(trunc_normal_weights(m.weight.size()))
            if m.bias is not None:
                m.bias.data.fill_(0.1)



