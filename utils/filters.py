import argparse, configparser
import re, os, sys
import numpy as np
import logging
from pydoc import locate
from utils.print_log import harddebug, printdebug


class ParamFilter():

    def __init__(self,
                 type=str,
                 interval=None,
                 values=None,
                 neg=False,
                 any_value=False,
                 always_true=False):

        assert bool(interval) + (values is not None) + any_value + always_true == 1

        self.type = type
        self.neg = neg
        self.is_interval = False
        self.any_value = False
        self.always_true = False
        
        if interval:
            self.is_interval=True
            self.interval=interval
            self.arg_str = 'in [' + '...'.join(str(_) for _ in interval) + ']'

        elif values is not None:
            self.values = values
            self.arg_str = 'in ' + ', '.join(str(_) for _ in values)

        elif any_value:
            
            self.any_value = True
            self.arg_str = 'any'
            
        elif always_true:
            self.always_true = True
            self.arg_str = 'always true'
            
        else:
            raise ValueError('Nothin given for filtering')

        if neg:
            self.arg_str = 'not ' + self.arg_str
            
    @classmethod
    def from_string(cls, arg_str='',
                    type=str):

        if arg_str is None:
            return cls(always_true=True, type=type)
        
        arg_str_ = arg_str.split()
        neg = bool(arg_str_) and arg_str_[0].lower() == 'not'

        interval = None
        values = None
        any_value = False
        always_true = False
        
        if neg:
            arg_str_ = arg_str_[1:]

        arg_str = ' '.join(arg_str_)

        interval_regex = '\.{2,}'
        is_interval = re.search(re.compile(interval_regex),
                                arg_str)

        list_regex = '[\s\,]+\s*'
        is_list = re.search(re.compile(list_regex),
                            arg_str)
        
        if is_interval:

            endpoints = re.split(interval_regex, arg_str)
            interval = [-np.inf, np.inf]
            for i in (0, -1):
                try:
                    interval[i] = type(endpoints[i])
                except ValueError:
                    pass

        elif is_list:

            values_str = re.split(list_regex, arg_str)
            values = [type(v) for v in values_str]

        elif not arg_str:
            any_value = True

        else:
            if type == bool:
                values = [arg_str.lower() == 'true']
            else:
                values = [type(arg_str)]

        return cls(type=type, interval=interval, values=values,
                   neg=neg, any_value=any_value, always_true=always_true)

    def __str__(self):

        return self.arg_str

    @printdebug(False)
    def filter(self, value):

        harddebug(self, value)
        if self.always_true:
            return not self.neg
        
        if type(value) is list:
            if self.neg:
                return np.all([self.filter(v) for v in value])
            else:
                return np.any([self.filter(v) for v in value])
                              
        if self.any_value:

            return isinstance(value, self.type) ^ self.neg
            
        if self.is_interval:
            try:
                a, b = self.interval
                in_ =  a <= value <= b
                return in_ ^ self.neg
            except TypeError as e:
                return self.neg
                logging.error('Wrong type filter:', a, type(a),
                              b, type(b),
                              value, type(value),
                              self)
                raise(e)
            
        in_ = value in self.values
        return in_ ^ self.neg


class ListOfParamFilters(list):

    def __init__(self, *a, fragile=False, **kw):
        super(list).__init__(*a, **kw)
        assert not fragile or len(self) <= 1
        self._fragile = fragile
    
    @property
    def type(self):
        if not self:
            return None
        return self[0].type

    @property
    def always_true(self):
        return all(_.always_true for _ in self)
    
    def append(self, a):

        if self.type and a.type != self.type:
            print('List type {} filter type {}'.format(self.type, a.type))
        assert not self.type or a.type == self.type
        if self._fragile and self:
            self[0] = a
            self._fragile = False
        else:
            super().append(a)
        
    def filter(self, value):

        return all(_.filter(value) for _ in self)

    def __str__(self):

        return ', '.join(str(_) for _ in self)
    
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


    def __str__(self):

        return '\n'.join('{k}: {f}'.format(k=k, f=self[k]) for k in self)
    
    
class FilterAction(argparse.Action):

    def __init__(self, option_strings, dest, type=str, **kwargs):
        super(FilterAction, self).__init__(option_strings, dest, **kwargs)
        self._of_type = type
        default_filter = ParamFilter.from_string(type=type, arg_str=self.default)
        self.default = ListOfParamFilters(fragile=True)
        self.default.append(default_filter)
        # print('$$$', default_filter.type, type)

        # print('FilterAction init', *option_strings, 'filter', str(self.default), 'for', type.__name__)
        
    def __call__(self, parser, namespace, values, option_string=None):

        # print('namespace', namespace)
        # print('values', values)
        # print('option_string', option_string)
        filter = ParamFilter.from_string(type=self._of_type, arg_str=' '.join(values))
        getattr(namespace, self.dest).append(filter)

def get_filter_keys(from_file=os.path.join('utils', 'filters.ini')):

    filters = configparser.ConfigParser()
    filters.read(from_file)

    types = dict(filters['type'])
    dest = dict(filters['dest'])

    return [dest.get(_, _) for _ in types]

        
if __name__ == '__main__':

    argv = '--my-int not 0'
    argv_ = argv.split()
    
    argv = argv_ if not sys.argv[0] else None
    
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--my-int', action=FilterAction, nargs='*', default='')
    parser.add_argument('--my-bool', action=FilterAction, nargs='*')

    args = parser.parse_args(argv)

    filters = DictOfListsOfParamFilters()

    for _ in args.__dict__:

        filters.add(_, args.__dict__[_])

    print(filters)

        
