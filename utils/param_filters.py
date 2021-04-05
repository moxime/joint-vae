import re
import numpy as np


class ParamFilter():

    def __init__(self, arg_str='',
                 arg_type=int,
                 neg=False,
                 always_true=False,
    ):

        self.arg_str = arg_str
        self.arg_type = arg_type
        self.always_true = always_true
        self.neg = neg
        
        interval_regex = '\.{2,}'
        self.is_interval = re.search(re.compile(interval_regex),
                                arg_str)

        list_regex = '[\s\,]+\s*'
        self.is_list = re.search(re.compile(list_regex),
                            arg_str)

        if self.is_interval:

            endpoints = re.split(interval_regex, arg_str)
            self.interval = [-np.inf, np.inf]
            for i in (0, -1):
                try:
                    self.interval[i] = arg_type(endpoints[i])
                except ValueError:
                    pass

        if self.is_list:

            _values = re.split(list_regex, arg_str)
            self.values = [arg_type(v) for v in _values]

    def __str__(self):

        pre = 'not ' if self.neg else ''

        if self.is_interval:
            return pre + '..'.join([str(a) for a in self.interval])

        if self.is_list:
            return pre + ' or '.join([str(v) for v in self.values])

        if self.arg_type is bool:
            return 'False' if self.neg else 'True'

        else:
            return pre + (self.arg_str if self.arg_str else 'any')
        
    def filter(self, value):

        if type(value) is list:
            if self.neg:
                return np.all([self.filter(v) for v in value])
            else:
                return np.any([self.filter(v) for v in value])
                              
        neg = self.neg
        if self.always_true:
            return False if neg else True
        
        if not self.arg_str:
            return not value if neg else value
        
        # if not value:
        #    return True
        
        if self.is_interval:
            if value is None:
                return False

            try:
                a, b = self.interval
                in_ =  a <= value <= b
                return not in_ if neg else in_
            except TypeError as e:
                logging.error('Wrong type filter:', a, type(a),
                              b, type(b),
                              value, type(value),
                              self)
                raise(e)
            
        if self.is_list:
            in_ = value in self.values
            return not in_ if neg else in_

        # else
        the_value = self.arg_type(self.arg_str)
        in_ = value == the_value
        return not in_ if neg else in_
    
