import csv
import operator
import itertools as it
import datetime
import statistics
from collections import defaultdict
import sys
from io import StringIO

ALLOWED_COL_PROPERTIES = ['dtype','long_name','col_print_length','key','aggregation_func']

def nunique(values: list):
    """Count of unique values"""
    return len(set(values))

def count(values: list):
    """Count non-missing values"""
    return len(x for x in values if x is not None)

agg_functions ={'nunique':nunique,'mean':statistics.mean,'sum':sum,'median':statistics.median,'min':min,'max':max,'std':statistics.stdev,'var':statistics.variance,'count':count,'len':len}

def pretty_string(string,color,length=None):
    """Trim to length and change color of provided string.

    Given string is first trimmed to requested length, then color
    formatting is applied and the result returned.
    Trimming is performed by keepting the first [length] characters.
    It is useful to trim strings before formatting because
    formatting is performed by adding special characters thus
    making the string's length more than the visible characters.

    Parameters
    ----------
    string : str
             String which will be returned trimmed and formatted.
    color : str
            Color to apply to the string. Available colors are
            red, green, yellow, blue, magenta, cyan.
    length : int
             Length to which the provided string will be trimmed
             be removing characters from the right. This length
             is for the visible characters only and does not count
             the special formatting characters that are added
             by this function.

    Example
    -------
    Given string 'lengthy_column_name' apply blue font and return
    string that is only 4 characters long i.e. "leng"
    
    >>> pretty_string('lengthy_column_name','blue',4)

    """
    colors_dict = {
        'red':31,
        'green':32,
        'yellow':33,
        'blue':34,
        'magenta':35,
        'cyan':36,
    }
    # This assures that only the visible characters are trimmed and not the whole string including formatting
    if length==None:
        return f"\033[{colors_dict[color]}m{string}\033[0m"
    else:
        return f"\033[{colors_dict[color]}m{string[:length]}\033[0m"
    
def is_iterable(obj):
    """Check whether obj is iterable."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def element_wise_comparison(func, list_1, list_2):
    """Compare list_1 and list_2 using func and return a list of Bool

    Takes Python lists or tuples and outputs Python lists. list_2 may be a scalar.

    """
    if not is_iterable(list_1):
        raise TypeError("list_1 must be of the type 'List'")
    if isinstance(list_2, (int, float, str, datetime.datetime)) :
        # Compare list list_1 to a value list_2
        return [func(x,list_2) for x in list_1]
    elif is_iterable(list_2):
        # Compare list to a list if their lengths are compatible
        if len(list_1) != len(list_2):
            raise ValueError("Lists have incompatible lengths")
        return [func(x,y) for x, y in zip(iter(list_1), iter(list_2))]
    else:
        raise TypeError("Can only compare against the types 'Int,' 'Float,' 'Str,' or 'List'")

def aggregate_ignore_none(iterable, aggregation_func):
    """
    Aggregate function that ignores None values.
    
    If all values are None, returns None.
    """
    filtered_values = [value for value in iterable if value is not None]
    if not filtered_values:
        return None
    return aggregation_func(filtered_values)

class Category:
    """Data format for categorical data

    INCOMPLETE.  Will include list of categories and dict for encoding.

    """
    def __init__(self,data):
        self.data = data
        return None
        
    def __repr__(self):
        return self.data
        
    def __format__(self,fmt):
        return f"{self.data:{fmt}}"

class DataColumn:
    """Represents a column of a DataFrame.

    Stores the column's values and additional metadata
    to describe column properties. DataColumn is subscriptable.
    See examples below.

    Attributes
    ----------
    data : list
           Ordered ist of values belonging to this column.
    dtype : data type
            The type of the data in this column. For example
            str, int, Category.
    long_name : str
                Long name of the column intended for human
                understanding. Long_names can be useful
                for interpreting each column as the names
                that arae printed by DataFrame by default
                are short names and should emphasize brevity
                over meaningfulness.
    col_print_length : int
                The lengh (in number of characters) used when
                printing this column. Column label and values
                will be truncated to fit this length. 
                Calculaated by considering the min and max
                column widths as well as the column name
                and all values in the column.
    key : bool
          Boolean value indicating whether this column
          is a key or not. Key columns may not have missing 
          values and are used by the DataFrame for aggregations.
          Key columns are ignored when DataFrame exports data
          for analysis by default.
    aggregation_func : callable
                A callable that must accept an iterable
                and return a single value. This is used 
                by DataFrame to aggregate the data.

    Methods
    -------
    apply(func)
    sum()
    min()
    max()
    mean()
    median()
    median_low()
    median_high()
    mode()
    std()
    var()
    pstd()
    pvariance()
    cov()
    cor()
    lr(other)
    set_type(new_type)
    isna()
    any_na()
    unique()

    Examples
    --------
    Create new column with four values:
    >>> col = DataColumn([0,9,8,7])
    Select first two elements from column's data:
    >>> col[:2]
    Returns list [0,9].

    """
    def __init__(self, data, col_properties:dict=None):
        """Initiate new column
        
        New column containing the provided data and properties,
        if provided. If no col_properties is passed, initiates 
        all column properties with the value of None.

        Parameters
        ----------
        data : list
               List of the column's values.
        col_properties : dict
                         Property: value pairs. 
        
        """
        self.data = data
        # Initiate empty properites
        for prop in ALLOWED_COL_PROPERTIES:
            self._set_properties({prop:None})
        # Set the passed properties
        if col_properties != None:
            self._set_properties(col_properties)
        return
            
    def _set_properties(self, property_dict):
        """Set column property"""
        if isinstance(property_dict, dict):
            for attr_name, attr_val in property_dict.items():
                if attr_name == 'aggregation_func':
                    if isinstance(attr_val,str):
                        setattr(self, attr_name, agg_functions[attr_val])
                    else:
                        setattr(self,attr_name,attr_val)
                else:
                    setattr(self, attr_name, attr_val)
        else:
            raise TypeError("property_dict parameter must be of the type 'Dict'")
        return

    def _get_property(self, property_name):
        """Extract a property value"""
        try:
            return getattr(self,property_name,None)
        except:
            raise ValueError(f"Property {property_name} not found")

    def _get_all_properties(self):
        """
        Extract all properties
        
        Form a dicct of dicts that can be used to recreate this column i.e. in DataColumn
        class initialization.
        
        """
        all_properties = {}
        for prop in ALLOWED_COL_PROPERTIES:
            all_properties[prop] = self._get_property(prop)
        return all_properties

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DataColumn([operator.add(x, other) for x in self.data])
        elif is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            return DataColumn([x + y for x, y in zip(iter(self),iter(other))])
        else:
            raise TypeError("Operands must be iterable or 'Int,' or 'Float'")

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DataColumn([operator.sub(x, other) for x in self.data])
        elif is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            return DataColumn([x - y for x, y in zip(iter(self),iter(other))])
        else:
            raise TypeError("Operands must be iterable or 'Int,' or 'Float'")

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return DataColumn([operator.mul(x, other) for x in self.data])
        elif is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            return DataColumn([x * y for x, y in zip(iter(self),iter(other))])
        else:
            raise TypeError("Operands must be iterable or 'Int,' or 'Float'")

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if other==0:
                raise ValueError("Div by zero is not allowed")
            return DataColumn([operator.truediv(x, other) for x in self.data])
        elif is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            if 0 in other:
                raise ValueError("Encountered division by zero")
            return DataColumn([x / y for x, y in zip(iter(self),iter(other))])
        else:
            raise TypeError("Can only divide by the types 'Int,' or 'Float'")

    def __eq__(self, other):
        return element_wise_comparison(operator.eq,self, other)

    def __lt__(self, other):
        return element_wise_comparison(operator.lt,self, other)

    def __le__(self, other):
        
        return element_wise_comparison(operator.le,self, other)

    def __ne__(self, other):
        
        return element_wise_comparison(operator.ne,self, other)

    def __ge__(self, other):
        
        return element_wise_comparison(operator.ge,self, other)

    def __gt__(self, other):
        
        return element_wise_comparison(operator.gt,self, other)
        
    def __repr__(self):
        # Redirect stdout to a StringIO object
        stdout_backup = sys.stdout
        sys.stdout = StringIO()
        # Call the method
        print(self.data[:5])
        # Get the captured output
        captured_output = sys.stdout.getvalue()
        # Restore stdout
        sys.stdout = stdout_backup
        return f"{captured_output}\nDataColumn of size {len(self)}"

    def as_list(self):
        """Return this column's values as a list"""
        return self.data

    def __iter__(self):
        return iter(self.data)

    def apply(self, func):
        """Map func onto this column's values"""
        return DataColumn(list(map(func,self.data)))

    def sum(self):
        """Return the sum of this column's values"""
        return sum(self.data)

    def min(self):
        """Return the smallest of this column's values"""
        return min(self.data)

    def max(self):
        """Return the largest of this column's values"""
        return max(self.data)

    def mean(self):
        """Return the mean of this column's values"""
        return statistics.mean(self.data)

    def median(self):
        """Return the median of this column's values"""
        return statistics.median(self.data)

    def median_low(self):
        """Return the low median of this column's values"""
        return statistics.median_low(self.data)

    def median_high(self):
        """Return the high median of this column's values"""
        return statistics.median_high(self.data)

    def mode(self):
        """Return the mode of this column's values"""
        return statistics.mode(self.data)

    def std(self):
        """Return the sample standard deviation of this column's values"""
        return statistics.stdev(self.data)

    def var(self):
        """Return the sample variance of this column's values"""
        return statistics.variance(self.data)

    def pstd(self):
        """Return the population standard deviation of this column's values"""
        return statistics.pstdev(self.data)

    def pvariance(self):
        """Return the population variance of this column's values"""
        return statistics.pvariance(self.data)

    def cov(self,other):
        """Return the covariance of this column with other column"""
        if isinstance(other, DataColumn):
            return statistics.covariance(self.data,other.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def cor(self,other):
        """Return the correlation of this column with other column"""
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

    def as_type(self, new_type):
        """Returns DataColumn equivalent to this but with values cast to new_type"""
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
        col_props = self._get_all_properties()
        return DataColumn(casted_values,col_properties=col_props)

    def isna(self):
        """Return list of bools indicating missing values"""
        return list(map(lambda x: x==None,self.data))

    def any_na(self):
        """Return True if any value is None"""
        return any(map(lambda x: x == None,self.data))

    def fillna(self,fill_val):
        """Return DataColumn with fill_value in place of missing values"""
        new_values = list(map(lambda x: fill_val if x==None else x,self.data))
        col_props = self._get_all_properties()
        return DataColumn(new_values,col_properties=col_props)

    def unique(self):
        return(list(set(self.data)))
        
class DataFrame:
    '''
    Simplistic DataFrame

    Consists of columns represented by DataColumn class.

    Functions
    ---------
    read_csv
    to_csv
    apply

    '''
    def __init__(self,data=None,dtypes=None,col_properties=None,row_index=None,row_index_labels=None):
        """
        Initiate new DataFrame, either empty or from values
        
        Returns empty DataFrame if data is None. If data is provided,
        populates the dataframe accordingly. If dtypes are provdied,
        casts and sets the values accordingly. If col_properties are
        provided, sets the properties accordingly.
        Since dtypes can also be specified in col_properties,
        if both dtypes and col_properties were given, returns error.

        Parameters
        ----------
        data : dict {str : iterable}
               Keys are used as names (short names) for the frame's
               columns. Defaults to None.
        dtypes : dict {str : type}
                 Datatypes to use for each column. Defaults to None.
        col_properties : dict {str : dict}
                         For each column, provided a dict of column
                         properties. Defaults to None.
        
        """
        dtypes_provided = isinstance(dtypes,dict)
        col_props_provided = isinstance(col_properties,dict)
        # Make sure that only one of dtypes and col_propperties was provided
        # since col_properties can include dtypes.
        if dtypes_provided and col_props_provided:
            raise TypeError("Can only specify one of the parameters dtypes and col_properties")
        values_len = -1
        # Set default values for internal parameters
        #self._default_col_print_length = 10
        self._max_col_print_length = 10
        self._min_col_print_length = 5
        self.rows = NestedDict(assume_sorted=True)
        self._data = []
        self.columns = {} # keys are short names; col_properties includes long_name
        self.row_index = NestedDict(assume_sorted=True)
        self.row_index_labels = []
        if data==None:
            pass
        elif isinstance(data,dict):
            col_idx = 0 # iterate column index
            for key, values in data.items():
                # Check column lengths compatibiilty
                if values_len == -1:
                    values_len = len(values)
                else:
                    if len(values) != values_len:
                        raise ValueError("Columns have incompatible lengths")
                # Check if dtypes were given and store data values
                if dtypes_provided:
                    self._data.append(DataColumn([dtypes[key](val) for val in values]))
                    self._data[col_idx]._set_properties({'dtype':dtypes[key]})
                elif col_props_provided:
                    self._data.append(DataColumn(values))
                    self._data[col_idx]._set_properties(col_properties[key])
                else:
                    self._data.append(DataColumn(values))
                # Add column to columns dict
                self.columns[key] = col_idx
                col_idx += 1
            self._update_col_lengths()
            self.rows = row_index
            self.row_index_labels = row_index_labels
        else:
            raise TypeError("Data must be of the type'Dict'")
        return

    def _update_col_lengths(self,col=None):
        """Update the printing width of the column"""
        if col==None:
            for col_label, col_idx in self.columns.items():
                new_length = min(self._max_col_print_length,max([len(str(x))+1 for x in self._data[col_idx]]))
                new_length = max(new_length,len(col_label)+1)
                new_length = max(new_length,self._min_col_print_length)
                self.set_property('col_print_length',{col_label:new_length})
            return
        else:
            col_label = col
            col_idx = self.columns[col_label]
            new_length = min(self._max_col_print_length,max([len(str(x))+1 for x in self._data[col_idx]]))
            new_length = max(new_length,len(col_label)+1)
            new_length = max(new_length,self._min_col_print_length)
            self.set_property('col_print_length',{col_label:new_length})
            return
            
    def read_csv(self, file_path):
        """
        Read data from a csv file.
        
        Populates this frame, if empty, with data read from the csv file file_path.
        Column headers from the file will be stored as the short column names. If you
        want to replace them, use set_short_col_names method of this frame in a consequent
        step.

        Parameters
        ----------
        file_path : str
                    Path to a csv file.
        
        """
        # Make sure this frame is empty
        if len(self.columns) > 0:
            raise RuntimeError("Attemped to overwrite current data with read_csv")
        # Read the file
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file,skipinitialspace=True) # https://docs.python.org/3/library/csv.html
            columns = next(csv_reader)
            for i, col_label in enumerate(columns):
                self.columns[col_label] = i
            data = []
            for row in csv_reader:
                processed_row = [None if value == '' else value for value in row]
                data.append(processed_row)
        self._data = list(zip(*data)) # transpose
        # Transform into DataColumn types
        for col_idx, col_data in enumerate(self._data):
            self._data[col_idx] = DataColumn(col_data)
        del data;
        self._update_col_lengths()
        return

    def to_csv(self, file_path):
        """
        Store this frame's data into a csv file
        
        INCOMPLETE, need to fix _data and address the questions in the comments.
        
        """
        with open(file_path, 'w', newline='') as file: # newline????
            csv_writer = csv.writer(file) # https://docs.python.org/3/library/csv.html
            csv_writer.writerow(list(self.columns.keys()))
            csv_writer.writerows(self._data)

    def __getitem__(self, key):
        """
        Select elements from the DataFrame

        Examples
        --------
        Assume the following DataFrame df:
          | col_a  | col_b  |  col_val   | 
        i |    UNK |    UNK |        UNK | 
        ----------------------------------
        0 |      1 |     10 |        0.2 | 
        1 |      1 |     10 |        0.3 | 
        2 |      1 |     10 |        0.5 | 
        3 |      1 |     10 |        0.4 | 
        4 |      1 |     10 |        0.3 | 
        ...
        
        Column selectors:
        df['col_a'] - column col_a
        To select multiple columns, use a list.  Do not use a tuple.
        df[['col_a','col_val']] - columns col_a, col_val

        Column and row selectors:
        df[:10, ['col_a','col_b']] - first 10 rows, columns col_a, col_b
        df[[1,4], ['col_a','col_b']] - rows 1 and 4, columns col_a, 
        df[[True,False,False,True], ] - rows 0 and 3, all columns
        df[[1,4], ['col_a',2]] - rows 1 and 4, columns col_a, col_val
        
        """
        all_cols_properties = {}
        if isinstance(key,tuple):
            new_data_dict = {} # Store the selected data, then use it to create and return new DataFrame
            new_cols = []
            use_all_cols = False
            # Extract row selector
            if isinstance(key[0],DataColumn):
                row_selector = key[0].data
            else:
                row_selector = key[0]
            # Extract column selector
            ## If empty, use all columns
            try:
                key[1]
            except:
                use_all_cols=True
            if use_all_cols:
                new_cols = list(self.columns.keys())
            elif isinstance(key[1],(list,tuple)):
                # if column selector is iterable, extract values into new_cols list
                new_cols = list(key[1])
            else:
                # otherwise create a list with just the one element
                new_cols.append(key[1])
            # Make sure that the new_list contains column labels, not indices
            for i, col in enumerate(new_cols):
                if isinstance(col,str):
                    pass
                elif isinstance(col,int):
                    new_cols[i] = list(self.columns.keys())[col]
                else:
                    raise TypeError("Column selector must contain str or int values.")
            # Extract data here
            ## For each selected column...
            for col_label, col_idx in self.columns.items():
                if col_label in new_cols:
                    if isinstance(row_selector,list):
                        if isinstance(row_selector[0],bool):
                            new_data_dict[col_label] = [x for x, is_selected in zip(self._data[col_idx],row_selector) if is_selected]
                        elif isinstance(row_selector[0],int):
                            new_data_dict[col_label] = [self._data[col_idx][x] for x in row_selector]
                    else:
                        new_data_dict[col_label] = self._data[col_idx][row_selector]
                    all_cols_properties[col_label] = self._data[col_idx]._get_all_properties()
            # Extract index info so it can be passed to the new frame
            new_row_index_labels = [col for col in self.row_index_labels if col in new_data_dict.keys()]
            return DataFrame(new_data_dict,col_properties=all_cols_properties,row_index_labels=new_row_index_labels) # Do not replicate the same row_index as in self because it may be based on columns that were not selected.  row_index_labels is ok
        elif isinstance(key, int):
            return DataColumn(self._data[key],col_properties=self._data[key]._get_all_properties())
        elif isinstance(key, str):
            try:
                col_idx = self.columns[key]
                return DataColumn(self._data[col_idx],col_properties=self._data[col_idx]._get_all_properties())
            except ValueError:
                raise KeyError(f"Column '{key}' not found")
        elif isinstance(key, list):
            new_data_dict = {} # Store the selected data, then use it to create and return new DataFrame
            # extract values into new_cols list
            new_cols = list(key)
            # Make sure that the new_list contains column indices, not labels
            for i, col in enumerate(new_cols):
                if isinstance(col,str):
                    new_cols[i] = self.columns[col]
                elif isinstance(col,int):
                    pass
                else:
                    raise TypeError("Column selector must contain str or int values.")
            # For each selected column...
            for col_label, col_idx in self.columns.items():
                if col_idx in new_cols:
                    new_data_dict[col_label] = self._data[col_idx]
                    all_cols_properties[col_label] = self._data[col_idx]._get_all_properties()
            # Extract index info so it can be passed to the new frame
            new_row_index_labels = [col for col in self.row_index_labels if col in new_data_dict.keys()]
            return DataFrame(new_data_dict,col_properties=all_cols_properties,row_index_labels=new_row_index_labels)

    def __setitem__(self, key, new_col_values):
        """
        Modify existing column or create new column.
        
        New values must be either DataColumn, list, int, float, str, datetime, or bool."""
        required_col_len = len(self._data[0])
        if isinstance(key, str):
            col_label = key
            # If exists, find the column index, otherwise check if possible (corrent length) to create the column
            if key in self.columns:
                # Column exists
                col_idx = self.columns[key]
            else:
                col_idx = len(self.columns) # b/c current length is 1 greater than current rightmost idn
                self.columns[key] = len(self.columns)
                self._data.append(DataColumn([None]*required_col_len))
        elif isinstance(key, int):
            col_idx = key
            col_label = list(self.columns.keys())[col_idx]
        else:
            raise TypeError("Key must be of the types 'Str' or 'Int'")
        if isinstance(new_col_values, DataColumn):
            if len(self._data[col_idx]) != len(new_col_values):
                raise ValueError("Columns have incompatible lengths")
            self._data[col_idx] = new_col_values
        elif isinstance(new_col_values, list):
            if len(self._data[col_idx]) != len(new_col_values):
                raise ValueError("Columns have incompatible lengths")
            # Get the properties of the column being replaced and create new column with the same properties
            col_props = self._data[col_idx]._get_all_properties()
            self._data[col_idx] = DataColumn(new_col_values,col_properties=col_props)
        elif isinstance(new_col_values,(int,str,float,bool,datetime,Category)):
            # Get the properties of the column being replaced and create new column with the same properties
            col_props = self._data[col_idx]._get_all_properties()
            self._data[col_idx] = DataColumn([new_col_values]*required_col_len,col_properties=col_props)
        else:
            raise TypeError("New column values must be a list, DataColumn, Str, Int, Bool, or Datetime.")
        self._update_col_lengths(col=col_label)
        return
        
    def __len__(self):
        return len(self._data[0])
    
    def __repr__(self):
        # Redirect stdout to a StringIO object
        stdout_backup = sys.stdout
        sys.stdout = StringIO()
        # Call the method
        self.show()
        # Get the captured output
        captured_output = sys.stdout.getvalue()
        # Restore stdout
        sys.stdout = stdout_backup
        return f"{captured_output}\nDataFrame with {len(self.columns)} columns and {len(self)} rows"

    def __iter__(self):
        return iter(self._data)

    def show(self,start_row=0,nrows=5,show_index=True):
        """Print the requested rows of data.

        Parameters
        ----------
        start_row : int
                    First row to be printed, default: 0.
        nrows : int
                How many rows to print in total, default: 5.
        show_index : bool
                     Whether to print index.
        """
        display_data = [] # each element to represent a row (instead of col as is in self._data
        prefix_extra_len = len(str(start_row+nrows))-1
        prefix_header1 = "| " 
        prefix_header2 = "| "
        prefix_header3 = "| "
        prefix_line = "--"
        prefix_data = "f'| '"
        # Prepare prefix
        if show_index:
            prefix_header1 = f"{' ':>{1+prefix_extra_len}} |"
            prefix_header2 = f"{' ':>{1+prefix_extra_len}} |"
            prefix_header3 = f"{'i':>{1+prefix_extra_len}} |"
            prefix_line = "-"*(3+prefix_extra_len)
            prefix_data="f'{data_idx:>{1+prefix_extra_len}} |'"
        # Slice rows
        for col in self._data:
            col = list(it.islice(col,start_row,start_row+nrows))
            display_data.append(col)
        # Transpose for  printing row by row
        display_data = list(zip(*display_data))
        # Print header
        ## Row 1 (short name)
        row_1_string = ""
        row_1_string += prefix_header1 + " "
        for col_label, col_idx in self.columns.items():
            col_width = self._data[col_idx].col_print_length
            row_1_string += f"{col_label:^{col_width}}" + ' | '
        print(row_1_string)
        ## Row 2 (dtypes)
        print(prefix_header2,end=' ')
        for col_label, col_idx in self.columns.items():
            try:
                dtype = self._data[col_idx].dtype
                col_width = self._data[col_idx].col_print_length
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
        ## Row 3 (aggregation summary)
        print()
        print(prefix_header3,end=' ')
        for col_label, col_idx in self.columns.items():
            col_width = self._data[col_idx].col_print_length
            # If this is a key column, indicate that, otherwise get the aggregation function's name
            if self._data[col_idx].key:
                agg_funct_string = "*"
            else:
                try:
                    agg_funct_string = self._data[col_idx].aggregation_func.__name__
                except AttributeError:
                    agg_funct_string = ""
            warning_string = " !" if sum(self._data[col_idx].isna())>0 else ""
            agg_funct_string = agg_funct_string[:(col_width-len(warning_string))] # shorten the string if necessary
            
            text_to_print = f"{agg_funct_string:>{col_width-len(warning_string)}}"+pretty_string(warning_string,'red')
            print(text_to_print,end = ' | ')
            #except:
            #    pass
        
        # Break line
        print("\n"+prefix_line+("-"*(len(row_1_string)-1-len(prefix_line))))
        # Print rows, one col at a time
        for r in range(len(display_data)):
            data_idx = r + start_row # used in eval(prefix_data)
            print(eval(prefix_data),end=' ')
            for col_idx, col_val in enumerate(display_data[r]):
                text_to_print = "" # text to print for the current column, formatted below
                col_width = self._data[col_idx].col_print_length
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
        return f"DataFrame with {len(self.columns)} columns and {len(self._data[0])} rows"

    def set_property(self,property_type,new_properties):
        """Set the values of a property for one or more columns.

        Parameters
        ----------
        property_type : str
                        Name of the property to be set/changed
        new_properties : dict
                         A dict of the form short_col_label : value
                         to be used as the mapping of new values for
                         the property for the column indicated
                         by the dict's key
        """
        # Check that this is a property that can be set/modified:
        if property_type not in ALLOWED_COL_PROPERTIES:
            raise ValueError(f"Property_type must be one of {ALLOWED_COL_PROPERTIES}")
        # Make sure that dict was passed
        if not isinstance(new_properties, dict):
            raise TypeError(f"New_properies must be a dict")
        # Iterate over dict and set each column's property to the value
        # Column properties are stored within this DataFrame, not with Column class
        for col_name, prop_value in new_properties.items():
            col_idx = self.columns[col_name]
            self._data[col_idx]._set_properties({property_type:prop_value})
            # In addition, if dtype was changed, cast the column into the new dtype
            if property_type == 'dtype':
                self._data[col_idx] = self._data[col_idx].as_type(prop_value)
        return

    def set_short_col_names(self,new_names,promote_current_to_long_names=False):
        """
        Set or update columns' short names.

        Set short names for DataFrame's columns as indicated by new_names. In addition,
        if promote_current_to_long_names is True, current short names will be promoted
        to long names. Short names are used when the DataFrame is printed, they should
        emphasize brevity over clarity.

        Parameters
        ----------
        new_names : dict
                    Dictionary of current and new names.
        promote_current_to_long_names : bool
                                        Whether to promote the current short names 
                                        to new long names. Default: False.

        """
        # Make sure none of the new names is not already taken
        for new_label in new_names.values():
            if new_label in self.columns.keys():
                raise ValueError(f"Column {new_label} already exists")
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
            # Sort column elements to the original order as indicated by the dict values (column indices)
            self.columns = dict(sorted(self.columns.items(), key=lambda item: item[1]))
            self._update_col_lengths()
        return

    def get_col_names(self):
        """Get dict of column names  (short : long)"""
        col_names_dict = {}
        for col_short_name, col_idx in self.columns.items():
            try:
                col_names_dict[col_short_name] = self.col_properties[col_idx].long_name
            except AttributeError:
                col_names_dict[col_short_name] = None
        return col_names_dict

    def set_row_index(self,key_col_labels,return_index=False):
        """Builds the rows property based on the list of keys key_col_labels.

        The resulting rows property can be accessed via selector by listing
        key values in their hierarchical order.

        Returns a DataFrame.

        Examples
        --------
        df = DataFrame({'col_a':['A','B','A','B'],'col_b':[0,0,1,1],'val':[1,2,3,4]})
        Calling set_row_index(['col_a','col_b']) where col_a has unique values
        'A' and 'B' and col_b has unique values 0 and 1
        """
        key_cols = {} # map for indices
        key_cols_idx = [] # list of indices' idx in data
        value_cols = {}
        value_cols_idx = []
        if not isinstance(key_col_labels, list):
            key_col_labels = [key_col_labels]
        # List out index labels and locations
        for col_label in key_col_labels:
            key_cols[col_label] = self.columns[col_label]
            key_cols_idx.append(self.columns[col_label])
        # List out data labels and locations
        for col_label, col_idx in self.columns.items():
            if col_label in key_cols:
                continue
            else:
                value_cols[col_label] = col_idx
                value_cols_idx.append(col_idx)
        # Sort data according to the provided keys
        sorted_data=list(zip(*sorted(zip(*self.values()),key=lambda x: [x[col] for col in key_cols_idx])))

        rows = NestedDict(assume_sorted=True)
        # Create index
        ## Iterate row at a time (i.e. iterate transposed data model)
        for i in range(len(self._data[0])):
            rows[[sorted_data[dim_value][i] for dim_value in key_cols_idx]] = i#[self._data[col][i] for col in range(len(self._data))]
        # Recreate DataFrame using the results of this method
        if not return_index:
            new_data = {}
            new_col_properties = {} # nested dictionary (dict for each column)
            col_idx = 0
            for key_col_label, old_col_idx in key_cols.items():
                col_data = sorted_data[old_col_idx]
                col_props = self._data[old_col_idx]._get_all_properties()
                col_props['key'] = True
                new_data[key_col_label] = col_data
                new_col_properties[key_col_label] = col_props
                col_idx += 1
            for value_col_label, old_col_idx in value_cols.items():
                col_data = sorted_data[old_col_idx]
                col_props = self._data[old_col_idx]._get_all_properties()
                col_props['key'] = False
                new_data[value_col_label] = col_data
                new_col_properties[value_col_label] = col_props
                col_idx +=1
            resulting_data_frame = DataFrame(new_data,col_properties=new_col_properties,row_index=rows,row_index_labels=key_col_labels)
            return resulting_data_frame
        else:
            return rows

    def aggregate(self,ignore_na=False):
        """Aggregates DataFrame using its keys.

        Keys must be set before aggregating. Aggregation_func property
        must also be set before aggregating.
        This method will reshape the data by creating one record 
        for each unique key combination. Values are aggregated using 
        aggregation_func property of each column.

        Parameters
        ----------
        ignore_na : bool
                    If set to True, will ignore any None values.
                    If an aggregation group contains only None values
                    for a column, that column's new value is set to None.
                    Defaults to False.
        """
        # Check for nulls
        if not ignore_na:
            for col_name, col_idx in self.columns.items():
                if self._data[col_idx].any_na():
                    raise ValueError(f"Missing value in column {col_name}")
        new_data = [[] for col in self.columns.items()]
        row_idx = 0
        # Iterate over all unique key combinations
        for key_values in self.rows.list_levels():
            # Aggregate values within this key combination, one column at a time
            for col_name, col_idx in self.columns.items():
                # Columns marked as keys are not aggregated -- they are the keys
                if self._data[col_idx].key:
                    current_value = self._data[col_idx][self.rows[key_values]][0]
                    new_data[col_idx].append(current_value)
                else:
                    # Apply this column's aggregation function
                    current_values = self._data[col_idx][self.rows[key_values]]
                    if ignore_na:
                        current_values = [value for value in current_values if value is not None]
                    new_value = self._data[col_idx].aggregation_func(current_values)
                    new_data[col_idx].append(new_value)
            self.rows[key_values] = row_idx
            row_idx += 1
        # Recreate dataframe one column at a time
        for col_label, col_idx in self.columns.items():
            new_col_data = new_data[col_idx]
            new_col_props = self._data[col_idx]._get_all_properties()
            self._data[col_idx] = DataColumn(new_col_data,col_properties=new_col_props)
        return

    def values(self, transpose=False, skip_col_labels=[], return_labels=False):
        """
        Returns a nested list of values.

        The returned nested list has shape n_cols x n_rows.

        Parameters
        ----------
        transpose : Boolean
                    If True, will transpose to n_rows x n_cols.
                    Default False.
        skip_col_labels : List of strings
                          List of which columns to exclude from
                          the output.  Defaults to [] (empty list).
        return_labels : Bool
                        Whether also to return list of labels of columns
                        which were actually returns. Default False

        Returns
        -------
        list if return_labels is False.
        (list, labels) if return_labels is True.
        """
        data_values = []
        data_labels = []
        for col_label, col_idx in self.columns.items():
            if not col_label in skip_col_labels:
                data_values.append(self._data[col_idx].data)
                data_labels.append(col_label)
        if transpose:
            data_values = list(zip(*data_values))
        if return_labels:
            return (data_values, data_labels)
        else:
            return data_values
        
    def dict(self):
        """
        Returns a dict that represents this data frame
        """
        data_dict = {}
        for col_label, col_idx in self.columns.items():
            data_dict[col_balel] = self._data[col_idx].data
        return data_dict

    def join(self, other):
        """
        Join other df to this using Left Join approach.
    
        Assumes that both frames have been indexed using the same key columns.
        """
        # Empty list for resulting data
        new_data = []
        # Transpose data to avoid repetitive slicing of the same indices
        l_full_data, l_col_labels = self.values(transpose=True, return_labels=True)
        r_full_data, r_col_labels = other.values(transpose=True, skip_col_labels=other.row_index_labels, return_labels=True) # key cols are already in l_full_data
        # Get indices
        l_index = self.rows
        r_index = other.rows
        joined_index = NestedDict(assume_sorted=True)
        # Helper:
        n_left_cols = len(l_col_labels)
        # Iterate overal all index keys and construct new data
        for keys in l_index.list_levels():
            l_slicer = l_index[*keys]
            l_data = l_full_data[l_slicer]
            l_len = len(l_data)
            r_slicer = r_index[*keys]
            if r_slicer == None:
                # Create dummy r_data while keeping correct number of columns
                r_data = [[None] for _ in r_col_labels]
                r_len = 1
            else:
                r_data = r_full_data[r_slicer]
                r_len = len(r_data)
            joined_data = [list(it.chain(x[0],x[1])) for x in list(it.product(l_data,r_data))]
            new_data.extend(joined_data)
        # Update l_slicer if records had to be repeated
        # Final preparation to construct the resulting DataFrame
        ## Transpose new_data to cols x rows
        new_data = list(zip(*new_data))
        new_data_dict = {}
        new_data_props = {}
        ## Iterate over new columns (two iterators, one over left, one over right)
        for i, col_label in enumerate(l_col_labels):
            new_data_dict[col_label] = new_data[i]
            new_data_props[col_label] = self[col_label]._get_all_properties()
        for i, col_label in enumerate(r_col_labels):
            new_data_dict[col_label] = new_data[i+n_left_cols]
            new_data_props[col_label] = other[col_label]._get_all_properties()
        return DataFrame(new_data_dict, col_properties=new_data_props)

class NestedDict:
    def __init__(self,assume_sorted:bool):
        if not isinstance(assume_sorted,bool):
            raise ValueError("assume_sorted must be provided as a bool")
        self.assume_sorted = assume_sorted
        self.data = defaultdict(lambda: None)
        return

    def __setitem__(self, keys, value):
        """Build nested dictionary from keys list and value object

        Each next key in keys will produce another nested dict key,
        with the last key's value being assigned the value
        as an element of a list.
        If the exact specified keys already exist, the value will
        be appended to the list value of the last key.
        """
        if len(keys)>1:
            if not keys[0] in self.data:
                self.data[keys[0]] = NestedDict(self.assume_sorted)
            self.data[keys[0]][keys[1:]] = value
        else:
            if not self.assume_sorted:
                # Collect list of indices
                if isinstance(self.data[keys[0]] ,list):
                    self.data[keys[0]].append(value)
                else:
                    self.data[keys[0]] = [value]
            else:
                # Build slicer objects
                if isinstance(self.data[keys[0]] ,slice):
                    self.data[keys[0]] = slice(self.data[keys[0]].start,value+1) # Replaces existing slice with new one by keeping the same start but modifying the end. This works because data is sorted
                else:
                    self.data[keys[0]] = slice(value,value+1)
        return

    def __getitem__(self, keys):
        if len(keys)>1:
            return self.data[keys[0]][keys[1:]]
        else:
            return self.data[keys[0]]

    def labels(self):
        return list(self.data.keys())

    def list_levels(self, trail=[]):
        """Generate list of allkey combinations.
        
        All, i.e. all levels', keys are returned as a nested list.
        Each inner list contains the keys for each key column.
        """
        resulting_levels=[]
        if isinstance(self.data[list(self.data.keys())[0]],NestedDict):
            for key, nested_dict in self.data.items():
                this_result = nested_dict.list_levels(trail=trail + [key])
                resulting_levels.extend(this_result)
        else:
            resulting_levels = [trail + [key] for key in self.data.keys()]
        return resulting_levels