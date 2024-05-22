import csv
import operator
import itertools as it
import datetime
import statistics
from collections import defaultdict

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

class ColumnProperties:
    def __init__(self,data=None):
        if data==None:
            return
        elif isinstance(data, dict):
            for attr_name, attr_val in data.items():
                setattr(self, attr_name, attr_val)
        else:
            raise TypeError("ColumnProperties data parameter must be of the type 'Dict'")

    def add_properties(self, data):
        if isinstance(data, dict):
            for attr_name, attr_val in data.items():
                setattr(self, attr_name, attr_val)
        else:
            raise TypeError("ColumnProperties data parameter must be of the type 'Dict'")

    def get_property(self, key):
        try:
            return self.key
        except:
            raise ValueError(f"Property {key} not found")

class Category:
    def __init__(self,data):
        self.data = data
        return None
        
    def __repr__(self):
        return self.data
        
    def __format__(self,fmt):
        return f"{self.data:{fmt}}"

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
        print(self.data[:5])
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
        return DataColumn(casted_values)

    def isna(self):
        return list(map(lambda x: x==None,self.data))
        
class DataFrame:
    '''
    Simplistic DataFrame

    Properties that must be maintained as columns are added, deleted, or moved:
    self.data
    self.columns
    self.col_properties

    Functions
    ---------
    read_csv
    to_csv
    apply
    '''
    def __init__(self,data=None,dtypes=None):
        dtypes_provided = isinstance(dtypes,dict)
        self.allowed_col_properties = ['dtype','long_name','col_print_length']
        self.default_col_print_length = 10
        self.max_col_print_length = 10
        self.min_col_print_length = 5
        self.data = []
        self.columns = {} # keys are short names; col_properties includes long_name
        self.col_properties = []
        values_len = -1
        if data==None:
            pass
        elif isinstance(data,dict):
            i = 0
            for key, values in data.items():
                if values_len == -1:
                    values_len = len(values)
                else:
                    if len(values) != values_len:
                        raise ValueError("Columns have incompatible lengths")
                self.columns[key] = i
                self.data.append(values)
                # Check if dtypes were given:
                if dtypes_provided:
                    self.col_properties.append(ColumnProperties({'dtype':dtypes[key]}))
                else:
                    self.col_properties.append(ColumnProperties())
                i += 1
            self.update_col_lengths()
        else:
            raise TypeError("Data must be of the type'Dict'")
        return

    def update_col_lengths(self,col=None):
        if col==None:
            for col_label, col_idx in self.columns.items():
                new_length = min(self.max_col_print_length,max([len(str(x))+1 for x in self.data[col_idx]]))
                new_length = max(new_length,len(col_label)+1)
                new_length = max(new_length,self.min_col_print_length)
                self.set_property('col_print_length',{col_label:new_length})
            return
        else:
            col_label = col
            col_idx = self.columns[col_label]
            new_length = min(self.max_col_print_length,max([len(str(x))+1 for x in self.data[col_idx]]))
            new_length = max(new_length,len(col_label)+1)
            new_length = max(new_length,self.min_col_print_length)
            self.set_property('col_print_length',{col_label:new_length})
            return
            
    def read_csv(self, file_path):
        if len(self.columns) > 0:
            raise RuntimeError("Attemped to overwrite current data with read_csv")
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file,skipinitialspace=True) # https://docs.python.org/3/library/csv.html
            columns = next(csv_reader)
            for i, col_label in enumerate(columns):
                self.columns[col_label] = i
            data = []
            for row in csv_reader:
                processed_row = [None if value == '' else value for value in row]
                data.append(processed_row)
            self.data = list(zip(*data))
        del data;
        self.col_properties = [ColumnProperties({'dtype':None}) for i in range(len(self.columns))]
        self.update_col_lengths()
        return

    def to_csv(self, file_path):
        with open(file_path, 'w', newline='') as file: # newline????
            csv_writer = csv.writer(file) # https://docs.python.org/3/library/csv.html
            csv_writer.writerow(list(self.columns.keys()))
            csv_writer.writerows(self.data)

    def __getitem__(self, key):
        if isinstance(key,(list,tuple)):
            new_data_dict = {}
            new_dtypes = {}
            new_cols = []
            use_all_cols = False
            # Handle row selections if rows were provided as a list (only bool and int values are accepted, may need to add explicit error handling)
            if isinstance(key[0],DataColumn):
                row_selector = key[0].data
            else:
                row_selector = key[0]
            # Decypher selected columns: first check if multipe or single; next, check whether labels or indices
            try:
                key[1]
            except:
                use_all_cols=True
            if use_all_cols:
                new_cols = list(self.columns.keys())
            elif isinstance(key[1],(list,tuple)):
                new_cols = list(key[1])
            else:
                new_cols.append(key[1])
            for i, col in enumerate(new_cols):
                if isinstance(col,str):
                    pass
                elif isinstance(col,int):
                    new_cols[i] = list(self.columns.keys())[i]
                else:
                    raise TypeError("Column selector must contain str or int values.")
            for col_label, col_idx in self.columns.items():
                if col_label in new_cols:
                    if isinstance(row_selector,list):
                        if isinstance(row_selector[0],bool):
                            new_data_dict[col_label] = [x for x, is_selected in zip(self.data[col_idx],row_selector) if is_selected]
                        elif isinstance(row_selector[0],int):
                            new_data_dict[col_label] = [self.data[x] for x in row_selector]
                    else:
                        new_data_dict[col_label] = self.data[col_idx][row_selector]
                    new_dtypes[col_label] = self.col_properties[col_idx].dtype
            #print(new_data_dict)
            #print(new_dtypes)
            return DataFrame(new_data_dict,dtypes=new_dtypes)
        elif isinstance(key, int):
            return DataColumn(self.data[key])
        elif isinstance(key, str):
            try:
                col_idx = self.columns[key]
                return DataColumn(self.data[col_idx])
            except ValueError:
                raise KeyError(f"Column '{key}' not found")

    def __setitem__(self, key, new_col_values):
        required_col_len = len(self.data[0])
        if isinstance(key, str):
            col_label = key
            # If exists, find the column index, otherwise check if possible (corrent length) to create the column
            if key in self.columns:
                # Column exists
                col_idx = self.columns[key]
            else:
                # Column does not exist
                if len(new_col_values.as_list()) != required_col_len:
                    ValueError("Columns have incompatible lengths")
                else:
                    col_idx = len(self.columns) # b/c current length is 1 greater than current rightmost idn
                    self.columns[key] = len(self.columns)
                    self.data.append([None]*required_col_len)
                    self.col_properties.append(ColumnProperties({'dtype':None}))
        elif isinstance(key, int):
            col_idx = key
            col_label = list(self.columns.keys())[col_idx]
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
        self.update_col_lengths(col=col_label)
        return
        
    def __len__(self):
        return len(self.data[0])
    
    def __repr__(self):
        # Human readable representation or informal, string, representation of the dataframe
        return str(self.show(start_row=0,nrows=5,show_index=True)) #str(list(self.data))

    def __iter__(self):
        return iter(self.data)

    def show(self,start_row=0,nrows=5,show_index=True):
        """Print nrows first rows of data
        """
        display_data = [] # each element to represent a row (instead of col as is in self.data
        prefix_extra_len = len(str(start_row+nrows))-1
        prefix_header1 = "| " 
        prefix_header2 = "| "
        prefix_line = "--"
        prefix_data = "f'| '"
        # Prepare prefix
        if show_index:
            prefix_header1 = f"{' ':>{1+prefix_extra_len}} |"
            prefix_header2 = f"{'i':>{1+prefix_extra_len}} |"
            prefix_line = "-"*(3+prefix_extra_len)
            prefix_data="f'{data_idx:>{1+prefix_extra_len}} |'"
        # Slice rows
        for col in self.data:
            col = list(it.islice(col,start_row,start_row+nrows))
            display_data.append(col)
        # Transpose for  printing row by row
        display_data = list(zip(*display_data))
        # Print header
        ## Row 1 (short name)
        row_1_string = ""
        row_1_string += prefix_header1 + " "
        for col_label, col_idx in self.columns.items():
            col_width = self.col_properties[col_idx].col_print_length
            row_1_string += f"{col_label:^{col_width}}" + ' | '
        print(row_1_string)
        ## Row 2 (dtypes)
        print(prefix_header2,end=' ')
        for col_prop in self.col_properties:
            try:
                dtype = col_prop.dtype
                col_width = col_prop.col_print_length
                text_to_print = ""
                if dtype==str:
                    text_to_print = pretty_string(f"{'str':>{col_width}}",'magenta')
                elif dtype==int or dtype==float:
                    text_to_print = pretty_string(f"{'num':>{col_width}}",'green')
                elif dtype==Category:
                    text_to_print = pretty_string(f"{'C':>{col_width}}",'yellow') ########################### Need to specify whether dummiefied already or not and how many cats
                else:
                    text_to_print = pretty_string(f"{'UNK':>{col_width}}",'red')
                print(text_to_print,end = ' | ')
            except:
                pass
        # Break line
        print("\n"+prefix_line+("-"*(len(row_1_string)-1-len(prefix_line))))
        # Print rows, one col at a time
        for r in range(len(display_data)):
            data_idx = r + start_row
            print(eval(prefix_data),end=' ')
            for col_idx, col_val in enumerate(display_data[r]):
                text_to_print = "" # text to print for the current column, formatted below
                col_width = self.col_properties[col_idx].col_print_length
                if isinstance(col_val,float):
                    text_to_print=f"{col_val:>{col_width},.1f}"
                elif col_val==None:
                    text_to_print = pretty_string(f"{'--':>{col_width}}",'red')
                elif isinstance(col_val,Category):
                    text_to_print=f"{col_val:>{col_width}}"
                elif isinstance(col_val,str):
                    text_to_print=f"{col_val[:col_width]:>{col_width}}"
                else:
                    text_to_print=f"{col_val:>{col_width}}"
                print(text_to_print,end = ' | ')
            print('')
        # Return descriptive string
        return f"DataFrame with {len(self.columns)} columns and {len(self.data[0])} rows"

    def set_property(self,property_type,new_properties):
        """Set each column's data type accordingly to types.

        Currently, only accepts dict.
        """
        if property_type not in self.allowed_col_properties:
            raise ValueError(f"Property_type must be one of {self.allowed_col_properties}")
        if not isinstance(new_properties, dict):
            raise TypeError(f"New_properies must be a dict")
        for col_name, prop_value in new_properties.items():
            col_idx = self.columns[col_name]
            self.col_properties[col_idx].add_properties({property_type:prop_value})
            if property_type == 'dtype':
                self[col_name] = self[col_name].set_type(prop_value)

    def set_short_col_names(self,new_names,promote_current_to_long_names=False):
        new_long_names = {}
        if not isinstance(new_names,dict):
            raise TypeError("new_names must be a dict")
        else:
            for cur_label, new_label in new_names.items():
                if promote_current_to_long_names:
                    new_long_names[new_label] = cur_label
                self.columns[new_label] = self.columns[cur_label]
                del self.columns[cur_label]
            if promote_current_to_long_names:
                self.set_property('long_name',new_long_names)
            self.columns = dict(sorted(self.columns.items(), key=lambda item: item[1]))
            self.update_col_lengths()
        return

    def get_col_names(self):
        col_names_dict = {}
        for col_short_name, col_idx in self.columns.items():
            try:
                col_names_dict[col_short_name] = self.col_properties[col_idx].long_name
            except AttributeError:
                col_names_dict[col_short_name] = None
        return col_names_dict

    def set_row_index(self,idx_col_labels):
        new_index_cols = {} # map for indices
        new_index_idx = [] # list of indices' idx in data
        new_data_cols = {}
        new_data_idx = []
        #data_col_order = [] # indices of the final data organized such that index cols are first
        # List out index labels and locations
        for col_label in idx_col_labels:
            new_index_cols[col_label] = self.columns[col_label]
            new_index_idx.append(self.columns[col_label])
        # List out data labels and locations
        for col_label, col_idx in self.columns.items():
            if col_label in new_index_cols:
                continue
            else:
                new_data_cols[col_label] = col_idx
                new_data_idx.append(col_idx)
        # Calculate column order
        #data_col_order = list(new_index_cols.values())+list(new_data_cols.values())

        self.nested_dict = NestedDict()
        # Iterate row at a time (i.e. iterate transposed data model)
        for data_row in zip(*self.data):
            self.nested_dict[[data_row[xid] for xid in new_index_idx]] = [data_row[xid] for xid in new_data_idx]
        return

class NestedDict:
    def __init__(self):
        self.data = defaultdict(lambda: None)
        return

    def __setitem__(self, keys, value):
        if len(keys)>1:
            if not keys[0] in self.data:
                self.data[keys[0]] = NestedDict()
            self.data[keys[0]][keys[1:]] = value
        else:
            self.data[keys[0]] = value
        return

    def __getitem__(self, keys):
        if len(keys)>1:
            return self.data[keys[0]][keys[1:]]
        else:
            return self.data[keys[0]]

    def labels(self):
        return list(self.data.keys())

    def enum_levels(self, trail=[]):
        resulting_levels=[]
        if isinstance(self.data[list(self.data.keys())[0]],NestedDict):
            for key, nested_dict in self.data.items():
                this_result = nested_dict.enum_levels(trail=trail + [key])
                resulting_levels.extend(this_result)
        else:
            resulting_levels = [trail + [key] for key in self.data.keys()]
        return resulting_levels


            


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