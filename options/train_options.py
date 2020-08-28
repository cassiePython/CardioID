from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_epoch', default=20, type=int, help='# training epochs')
        self._parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for adam')
        self._parser.add_argument('--w_decay', type=float, default=1e-3, help='learning rate for adam')
        
        self._parser.add_argument('--w_similarity', type=float, default=1.0, help='weight for similarity loss')
        self._parser.add_argument('--w_classify', type=float, default=1.0, help='weight for classify loss')
        

        self.is_train = True
