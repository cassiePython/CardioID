from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        
       
        self._parser.add_argument('--result_file', type=str, default='results.txt', help='file containing results')
        
        self.is_train = False
