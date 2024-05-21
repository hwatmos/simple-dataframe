import csv
import operator
import itertools as it
import datetime
import statistics

def pretty_string(string,color,length=None):
    """Accepted colors: red, green, yellow, blue, magenta, cyan"""
    colors_dict = {
        'red':31,
        'green':32,
        'yellow':33,
        'blue':34,
        'magenta':35,
        'cyan':36,
    }
    if length==None:
        return f"\033[{colors_dict[color]}m{string}\033[0m"
    else:
        return f"\033[{colors_dict[color]}m{string[:length]}\033[0m"

def element_wise_comparison(func, list_1, list_2):
    """Compare list_1 and list_2 using func and return a list of Bool

    Takes Python lists or tuples and outputs Python lists. list_2 may be a scalar.
    """
    if not isinstance(list_1,(list,tuple)):
        raise TypeError("list_1 must be of the type 'List'")
    if isinstance(list_2, (int, float, str, datetime.datetime)) :
        return [func(x,list_2) for x in list_1]
    elif isinstance(list_2, (list,tuple)):
        if len(list_1) != len(list_2):
            raise ValueError("Lists have incompatible lengths")
        return [func(x,y) for x, y in zip(list_1, list_2)]
    else:
        raise TypeError("Can only compare against the types 'Int,' 'Float,' 'Str,' or 'List'")

class DataColumn:
    """
    Column of Simplistic DataFrame
    """
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DataColumn([operator.add(x, other) for x in self.data])
        elif isinstance(other, DataColumn):
            if len(self.data) != len(other.data):
                raise ValueError("Columns have incompatible lengths")
            return DataColumn([operator.add(x, y) for x, y in zip(self.data, other.data)])
        else:
            raise TypeError("Operands must be of the type 'DataColumn,' 'Int,' or 'Float'")

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DataColumn([operator.sub(x, other) for x in self.data])
        elif isinstance(other, DataColumn):
            if len(self.data) != len(other.data):
                raise ValueError("Columns have incompatible lengths")
            return DataColumn([operator.sub(x, y) for x, y in zip(self.data, other.data)])
        else:
            raise TypeError("Operands must be of the types 'DataColumn,' 'Int,' or 'Float'")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DataColumn([operator.mul(x, other) for x in self.data])
        elif isinstance(other, DataColumn):
            if len(self.data) != len(other.data):
                raise ValueError("Columns have incompatible lengths")
            return DataColumn([operator.mul(x, y) for x, y in zip(self.data, other.data)])
        else:
            raise TypeError("Can only divide by the types 'Int,' or 'Float'")

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if other==0:
                raise ValueError("Div by zero is not allowed")
            return DataColumn([operator.truediv(x, other) for x in self.data])
        elif isinstance(other, DataColumn):
            if len(self.data) != len(other.data):
                raise ValueError("Columns have incompatible lengths")
            if 0 in other:
                raise ValueError("Encountered division by zero")
            return DataColumn([operator.truediv(x, y) for x, y in zip(self.data, other.data)])
        else:
            raise TypeError("Can only divide by the types 'Int,' or 'Float'")

    def __eq__(self, other):
        list_1 = self.data
        list_2 = other.data if isinstance(other,DataColumn) else other
        return DataColumn(element_wise_comparison(operator.eq,list_1, list_2))

    def __lt__(self, other):
        list_1 = self.data
        list_2 = other.data if isinstance(other,DataColumn) else other
        return DataColumn(element_wise_comparison(operator.lt,list_1, list_2))

    def __le__(self, other):
        list_1 = self.data
        list_2 = other.data if isinstance(other,DataColumn) else other
        return DataColumn(element_wise_comparison(operator.le,list_1, list_2))

    def __ne__(self, other):
        list_1 = self.data
        list_2 = other.data if isinstance(other,DataColumn) else other
        return DataColumn(element_wise_comparison(operator.ne,list_1, list_2))

    def __ge__(self, other):
        list_1 = self.data
        list_2 = other.data if isinstance(other,DataColumn) else other
        return DataColumn(element_wise_comparison(operator.ge,list_1, list_2))

    def __gt__(self, other):
        list_1 = self.data
        list_2 = other.data if isinstance(other,DataColumn) else other
        return DataColumn(element_wise_comparison(operator.gt,list_1, list_2))
        
    def __repr__(self):
        print(repr(self.data))
        return "Column"

    def as_list(self):
        return self.data

    def __iter__(self):
        return iter(self.data)

    def apply(self, func):
        return DataColumn(list(map(func,self.data)))

    def min(self):
        return min(self.data)

    def max(self):
        return max(self.data)

    def mean(self):
        return statistics.mean(self.data)

    def median(self):
        return statistics.median(self.data)

    def median_low(self):
        return statistics.median_low(self.data)

    def median_high(self):
        return statistics.median_high(self.data)

    def mode(self):
        return statistics.mode(self.data)

    def std(self):
        return statistics.stdev(self.data)

    def var(self):
        return statistics.variance(self.data)

    def pstd(self):
        return statistics.pstdev(self.data)

    def pvariance(self):
        return statistics.pvariance(self.data)

    def cov(self,other):
        if isinstance(other, DataColumn):
            return statistics.covariance(self.data,other.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def cor(self,other):
        if isinstance(other, DataColumn):
            return statistics.correlation(self.data,other.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def lr(self,other):
        """Linear regression against another column.

        Regress this column on another column and return slope and intercept.
        https://docs.python.org/3/library/statistics.html

        Returns slope, intercept
        """
        if isinstance(other, DataColumn):
            return statistics.linear_regression(other.data,self.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def set_type(self, new_type):
        casted_values = []
        for val in self.data:
            try:
                if val==None:
                    casted_val=None
                else:
                    casted_val = new_type(val)
                casted_values.append(casted_val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannon cast {val} to {new_type}: {e}")
        return casted_values

class DataFrame:
    '''
    Simplistic DataFrame

    Functions
    ---------
    read_csv
    to_csv
    apply
    '''
    def __init__(self,data=None):
        self.data = []
        self.columns = []
        values_len = -1
        if data==None:
            return
        elif isinstance(data,dict):
            for key, values in data.items():
                if values_len == -1:
                    values_len = len(values)
                else:
                    if len(values) != values_len:
                        raise ValueError("Columns have incompatible lengths")
                self.columns.append(key)
                self.data.append(values)
        else:
            raise TypeError("Data must be of the type'Dict'")

    def read_csv(self, file_path):
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file,skipinitialspace=True) # https://docs.python.org/3/library/csv.html
            self.columns = next(csv_reader)
            data = []
            for row in csv_reader:
                processed_row = [None if value == '' else value for value in row]
                data.append(processed_row)
            self.data = list(zip(*data))
            del data;

    def to_csv(self, file_path):
        with open(file_path, 'w', newline='') as file: # newline????
            csv_writer = csv.writer(file) # https://docs.python.org/3/library/csv.html
            csv_writer.writerow(self.columns)
            csv_writer.writerows(self.data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return DataColumn(self.data[key])
        elif isinstance(key, str):
            try:
                col_idx = self.columns.index(key)
                return DataColumn(self.data[col_idx])
            except ValueError:
                raise KeyError(f"Column '{key}' not found")

    def __setitem__(self, key, new_col_values):
        required_col_len = len(self.data[0])
        if isinstance(key, str):
            # If exists, find the column index, otherwise check if possible (corrent length) to create the column
            if key in self.columns:
                col_idx = self.columns.index(key)
            else:
                if len(new_col_values.as_list()) != required_col_len:
                    ValueError("Columns have incompatible lengths")
                else:
                    col_idx = len(self.columns) # b/c current length is 1 greater than current rightmost idn
                    self.columns.append(key)
                    self.data.append([None]*required_col_len)
        elif isinstance(key, int):
            col_idx = key
        else:
            raise TypeError("Key must be of the types 'Str' or 'Int'")
        if isinstance(new_col_values, DataColumn):
            if len(self.data[col_idx]) != len(new_col_values.as_list()):
                raise ValueError("Columns have incompatible lengths")
            self.data[col_idx] = new_col_values.as_list()
        elif isinstance(new_col_values, list):
            if len(self.data[col_idx]) != len(new_col_values):
                raise ValueError("Columns have incompatible lengths")
            self.data[col_idx] = new_col_values
        return
        
    def __len__(self):
        return len(self.data[0])
    
    def __repr__(self):
        # Human readable representation or informal, string, representation of the dataframe
        return str(self.rows(start_row=0,nrows=5,show_index=True)) #str(list(self.data))

    def __iter__(self):
        return iter(self.data)

    def rows(self,start_row=0,nrows=5,show_index=True):
        """Print nrows first rows of data
        """
        display_data = [] # each element to represent a row (instead of col as is in self.data
        col_width = 10
        prefix_extra_len = len(str(start_row+nrows))-1
        prefix_header = "| " # prefix to print and the beginning of each row
        prefix_line = "--"
        prefix_data = "f'| '"
        # Prepare prefix
        if show_index:
            prefix_header = f"{'i':>{1+prefix_extra_len}} |"
            prefix_line = "-"*(3+prefix_extra_len)
            prefix_data="f'{data_idx:>{1+prefix_extra_len}} |'"
        # Slice rows
        for col in df:
            col = list(it.islice(col,start_row,start_row+nrows))
            display_data.append(col)
        # Transpose for  printing row by row
        display_data = list(zip(*display_data))
        # Print header
        print(prefix_header,end=' ')
        for c in self.columns:
            print(f"{c:^{col_width}}",end = ' | ')
        print("\n"+prefix_line+("-"*len(self.columns)*13))
        # Print rows, one col at a time
        for r in range(len(display_data)):
            data_idx = r + start_row
            print(eval(prefix_data),end=' ')
            for c in display_data[r]:
                c_print = "" # text to print for the current column, formatted below
                if isinstance(c,float):
                    c_print=f"{c:>{col_width},.1f}"
                elif c==None:
                    c_print = pretty_string(f"{'--':>{col_width}}",'red')
                else:
                    c_print=f"{c:>{col_width}}"
                print(c_print,end = ' | ')
            print('')
        # Return descriptive string
        return f"DataFrame with {len(self.columns)} columns and {len(self.data[0])} rows"

    def set_types(self,types):
        """Set each column's data type accordingly to types.

        Currently, only accepts dict.
        """
        if isinstance(types, dict):
            for col_lab, col_type in dict.items():
                return DataColumn(self[col_lab].set_type(col_type))
        else:
            raise TypeError("Types param must be of the type 'Dict'")


            


'''
https://docs.python.org/3/tutorial/classes.html

This style of access is clear, concise, and convenient. The use of iterators pervades and unifies Python. Behind the scenes, the for statement calls iter() on the container object. The function returns an iterator object that defines the method __next__() which accesses elements in the container one at a time. When there are no more elements, __next__() raises a StopIteration exception which tells the for loop to terminate. You can call the __next__() method using the next() built-in function; this example shows how it all works.

Having seen the mechanics behind the iterator protocol, it is easy to add iterator behavior to your classes. Define an __iter__() method which returns an object with a __next__() method. If the class defines __next__(), then __iter__() can just return self:

class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]
'''