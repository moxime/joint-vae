import argparse, configparser
import re
import numpy as np
import logging
from pydoc import locate
from utils.print_log import harddebug, printdebug


class ParamFilter():

    def __init__(self, arg_str='',
                 arg_type=int,
                 neg=False,
                 always_true=False):

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

    @printdebug(False)
    def filter(self, value):

        harddebug(self, value)
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
                return False
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


class ListOfParamFilters(list):

    @property
    def arg_type(self):
        if not self:
            return None
        return self[0].arg_type
    
    def append(self, a):

        assert not self.arg_type or a.arg_type == self.arg_type
        super().append(a)
        
    def filter(self, value):

        return all(_.filter(value) for _ in self)

    
class DictOfListsOfParamFilters(dict):

    def add(self, key, filter):

        if key not in self:
            self[key] = ListOfParamFilters()
        self[key].append(filter)

    @printdebug(False)
    def filter(self, d):
        harddebug(d)
        for k in self:
            if k in d and not self[k].filter(d[k]):
                return False

        return True

    
class FilterAction(argparse.Action):

    def __init__(self, option_strings, dest, of_type=str, neg=False, **kwargs):
        super(FilterAction, self).__init__(option_strings, dest, **kwargs)

        # print('FilterAction init', option_strings)
        self._type=of_type
        self.default=ParamFilter()

    def __call__(self, parser, namespace, values, option_string=None):

        # print('FilterAction called', option_string, values)

        if not values:
            values = []
        if type(values) is not list:
            values = [values]

        neg = False

        if values and values[0].lower() == 'not':
            neg = True
            values.pop(0)

        arg_str = ' '.join(str(v) for v in values)

        filter = ParamFilter(arg_str,
                             arg_type=self._type,
                             neg=neg)
        # print(filter)
        if not hasattr(namespace, 'filters'):
            setattr(namespace, 'filters', DictOfListsOfParamFilters())
        if self.dest not in namespace.filters:
            namespace.filters[self.dest] = ListOfParamFilters()
        namespace.filters[self.dest].append(filter)

        
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--bogus')
    a, ra = parser.parse_known_args()

    filter_parser = parse_filters(parents=[parser], add_help=True)
    
    filters = filter_parser.parse_args(ra).filters

    for k in filters:
        print(k, *filters[k])

