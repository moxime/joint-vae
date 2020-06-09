from cvae import ClassificationVariationalNetwork




class VariationalInformationBottleneck(ClassificationVariationalNetwork):

    predict_methods = ['direct']
    
    def __init__(self, *a, decoder_layer_sizes, **kw):
        
        super().__init__(*a, decoder_layer_sizes=[1],
                         **kw)

        for p in self.decoder.parameters():
            p.requires_grad_(False)


    def evaluate(self, x, **kw):

        return super().evaluate(x, repeat_input=False, **kw)

    def predict_after_evaluate(self, y_est, losses, method='direct'):

        if method == 'direct':
            return y_est.argmax(-1)

    def print_architecture(self, *a, excludes=(), **kw):

        super().print_architecture(*a,
                                   excludes=tuple(excludes)+('decoder',),
                                   **kw)


    def train(self, *a, **kw):

        beta = self.beta
        super().train(*a,
                      mse_loss_weight=0,
                      x_loss_weight=1,
                      kl_loss_weight=beta,
                      **kw)
