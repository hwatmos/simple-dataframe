"""
Module Simple Dataframe.

Project intended to provide a module for performing basic data science tasks
while maintaining a simplistic code-base that relies only on standard Python
libraries.  Transparency, clarity, simplicity are the focus.

"""

import csv
import operator
import itertools as it
import datetime
import statistics
from collections import defaultdict
import sys
from io import StringIO
from numbers import Number
from typing import Callable, Any, Iterable, Self, Iterator

# =========================================================================
# Helper Functions
# =========================================================================

def nunique(values: list, include_na: bool=True) -> int:
    """Count of unique values, including NA by default"""
    if include_na:
        return len(set(values))
    else:
        my_set=set(values)
        my_set.discard(None)
        return len(my_set)

def count(values: list) -> int:
    """Count non-missing values"""
    return len(x for x in values if x is not None)

def _pretty_string(string: str, color: str, length: int=None) -> str:
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

    Returns
    -------
    Decorated (colored) string.

    Example
    -------
    Given string 'lengthy_column_name' apply blue font and return
    string that is only 4 characters long i.e. "leng"
    
    >>> _pretty_string('lengthy_column_name','blue',4)

    """
    # This assures that only the visible characters are trimmed and not the whole string including formatting
    if length==None:
        return f"\033[{_colors_dict[color]}m{string}\033[0m"
    else:
        return f"\033[{_colors_dict[color]}m{string[:length]}\033[0m"
    
def _is_iterable(obj: any) -> bool:
    """Check whether obj is iterable and return True/False."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False
    
def _element_wise_comparison(func: Callable, list_1: list[Any], list_2: list[Any]) -> list[bool]:
    """Compare list_1 and list_2 using func and return a list of Bool

    Takes Python lists, tuples, or DataColumns and outputs Python lists. list_2 may be a scalar.

    """
    if not _is_iterable(list_1):
        raise TypeError("list_1 must be of the type 'List'")
    if isinstance(list_2, (int, float, str, datetime.datetime)) :
        # Compare list list_1 to a value list_2
        return [func(x,list_2) for x in list_1]
    elif _is_iterable(list_2):
        # Compare list to a list if their lengths are compatible
        if len(list_1) != len(list_2):
            raise ValueError("Lists have incompatible lengths")
        return [func(x,y) for x, y in zip(iter(list_1), iter(list_2))]
    else:
        raise TypeError("Can only compare against the types 'Int,' 'Float,' 'Str,' or 'List'")

def _calc_col_print_length(col_data: list[Any],col_label: str) -> int:
    """Calculate how many characters are required to print column, based on its data and label"""
    if isinstance(col_label,str):
        new_len = max(len(col_label)+1, max([len(str(x))+1 for x in col_data]))
    else:
        new_len = max([len(str(x))+1 for x in col_data])
    new_len = min(new_len, _MAX_COL_PRINT_LEN)
    new_len = max(new_len,_MIN_COL_PRINT_LEN)
    return new_len

def _cast_list(values: Any, new_type: Callable) -> Any:
    """Cast all values in a list to new_type. Return None for None."""
    casted_values = []
    for val in values:
        try:
            if val==None:
                casted_val=None
            elif isinstance(val, str) and issubclass(new_type, Number):
                casted_val = new_type(val.replace(",",""))
            else:
                casted_val = new_type(val)
            casted_values.append(casted_val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannon cast {val} to {new_type}: {e}")
    return casted_values

# Unused currently, saved for future use
#def _aggregate_ignore_none(iterable, aggregation_func):
#    """
#    Aggregate function that ignores None values.
#    
#    If all values are None, returns None.
#    """
#    filtered_values = [value for value in iterable if value is not None]
#    if not filtered_values:
#        return None
#    return aggregation_func(filtered_values)

# =========================================================================
# Helper Objects
# =========================================================================

_MAX_COL_PRINT_LEN = 16
_MIN_COL_PRINT_LEN = 5

ALLOWED_COL_PROPERTIES = [ # Attributes that describe an instance of the DataColumn class.  They can be set and read using DataColumn's methods
    'dtype', # datatype of the values stored in the column
    'short_name', # short name for the column.
    'long_name', # long name of the column. This is generally not used by the DataFrame for printing, instead DataFrame uses short names which are ALSO stored in the DataFrame itself.
    'col_print_length', # how many characters are required to display this column.
    'key', # whether this column should be considered as the "key."  Key columns are used by the DataFrame class to index data, i.e. a DataFrame's index is built from key columns.  Indices are used to aggregate data (i.e. they are the dimension by which the data is aggregated).  They are also used to join two DataFrame objects together (their indices must be created in the same manner i.e. using the same key columns in the same order via DataFrame method set_row_index())
    'aggregation_func', # function definition to use for aggregating this column's data.  Required for all non-key columns.
    'length', # how many values are in the column
]

COMPLEX_COL_PROPERTIES = [ # List of those column properties which require their individual custom processing logic
    'dtype',
    'col_print_length', # must be calculated by DataColumn initiator
    'aggregation_func', # must be checked whether a str or a def was used to set the property
    'length', # must be calculated by DataColumn initiator itself
]

# _agg_functions dict is used by DataColumn when aggregation_func attribute is being set using a string.  In that case, _agg_functions is used to get definition of the appropriate function.
_agg_functions ={
    'nunique':nunique,
    'mean':statistics.mean,
    'sum':sum,
    'median':statistics.median,
    'min':min,
    'max':max,
    'std':statistics.stdev,
    'var':statistics.variance,
    'count':count,
    'len':len
}

_colors_dict = {
    'red':31,
    'green':32,
    'yellow':33,
    'blue':34,
    'magenta':35,
    'cyan':36,
}

# =========================================================================
# Helper Classes
# =========================================================================

# Experimental _IndexSlice class planned for future use with _DataIndex to implement
# a custom, mutable, slicer with in-place addition operator.
# (Needed so that _DataIndex assignment operation is a true assignment instead
# of the current append operation.)
#class _IndexSlice:
#    """
#    Mutable slicer.
#
#    Returns slicer when called.
#    """
#    def __init__(self, start, stop):
#        self.start = start
#        self.stop = stop
#        return
#    def __add__(self, other):
#        if not isinstance(other, int):
#            raise TypeError("Unsupported type for addition")
#        self.stop += other
#        return self
#
#    # Experimental shorthands for modifying the self.stop, saved for potential future use
#    #def __lshift__(self, other):
#    #    if not isinstance(other, int):
#    #        raise TypeError("Unsupported type for addition")
#    #    self.stop = other
#    #    return
#
#    def __setitem__(self, slicer_attribute, new_value):
#        if slicer_attribute == 'stop':
#            self.stop = new_value
#        elif slicer_attribute == 'start':
#            self.start = new_value
#        else:
#            raise ValueError(f"Trying to set attrubute {slicer_attribute} which does not exist")
#
#    def __call__(self):
#        return slice(self.start, self.stop)

class _DataIndex:
    """
    Class that builds the index used for addressing rows of DataColumns stored in DataFrame

    Warning!
    --------
    _DataIndex implements an unusual assignment operator
    via __setitem__ which adds on the passed value dest_row_idx
    to the existing slicer or list. This behavior may be updated
    in the future.
    --------

    Uses the defaultdict class from Python standard library "collections" which acts
    like a regular Python dict but returns default value instead of raising errors
    when a requested key does not exist.
    The __init__ method initiates and instance of an empty index. Assignment operation
    must be performed for each combination of keys in order to build the index. See example
    below.

    Note
    ----
    The initiator takes a required parameter assume_sorted.  If True, the index is created
    using slices, otherwise using lists of row indices.  It is recommended that the data
    is sorted so that assume_sorted can be used for efficiency gain.
    Note that the set_row_index function sorts the data before creating index.

    Methods
    -------
    labels() : returns a list of labels used at the selected index level. For example,
               df.rows.labels() will display possible values at the first level of the index
               while df.rows['A'].labels() will display possible values at the second
               level but only values that are available under 'A.'
    list_levels() : returns a nested list representing all existing index key combinations.

    Examples
    --------
    Create new DataFrame and set index using two columns, col_a and col_b
        >>> data = {
        >>>     'col_a': ['A','A','B','C'],
        >>>     'col_b': ['a','b','a','a'],
        >>>     'col_val': [45,46,22,91]
        >>> }
        >>> df = DataFrame(data).set_row_index(['col_a','col_b'])
    This created the "rows" attribute which is the DataFrame df', row index.
    View all possible key combinations in this index:
        >>> df.rows.list_levels()
        [['A', 'a'], ['A', 'b'], ['B', 'a'], ['C', 'a']]
    Identify rows of data where col_a=='A' and col_b=='b':
        >>> df.rows['A','b']
        slice(1, 2, None)
    Get list of vales available at the second level under first level value 'A':
        >>> df.rows['A'].labels()
        ['a', 'b']
    
    """
    def __init__(self,assume_sorted: bool,index_desc: dict=None) -> None:
        """
        Initiates empty instance of index.

        If assume_sorted is True, indices will be stored as slicers.
        Otherwise, indices will be stored as lists of row numbers.

        If index_desc is passed, reconstruct index using it.

        """
        if not isinstance(assume_sorted,bool):
            raise ValueError("assume_sorted must be provided as a bool")
        self.assume_sorted = assume_sorted
        self.data = defaultdict(lambda: None)
        # Reconstruct index if index_desc was given
        if isinstance(index_desc, dict):
            for keys, slicer in index_desc.items():
                self[keys] = slicer
        return

    def __setitem__(self, keys: list[Any], dest_row_idx: int) -> None:
        """
        Builds the nested index for identifying rows in the data.

        Warning!
        --------
        _DataIndex implements an unusual assignment operator
        via __setitem__ which adds on the passed value dest_row_idx
        to the existing slicer or list.  This behavior may be updated
        in the future.
        --------

        The keys list contains the consecutive index levels' values
        that point to the data row indicated by dest_row_idx.
        Specifically:
        Each next key in keys will produce another nested dict key.
        The last key will produce the inner-most dict where values
        are either slicers or lists of indices (depending
        on the assume_sorted attribute of this index) constructed
        from the value of the dest_row_idx parmeter.

        Parameters
        ----------
        keys : list
               List of values where each value corresponds 
               to the next consecutive index level. For example
               ['col_1_value','col_2_value'].
        dest_row_idx : int
                       Integer indicating row number of data
                       belonging to this combination of index
                       keys.

        Examples
        --------
        Create df:
            >>> data = {
            >>>     'col_a': ['A','A','A','B','C'],
            >>>     'col_b': ['a','a','b','a','a'],
            >>>     'col_val': [45,22,46,22,91]
            >>> }
            >>> df = DataFrame(data)
        Create index:
            >>> df.rows = _DataIndex(assume_sorted=True)
        Add key ['A','a'] to the index and assign the two rows
        that have these values in columns col_a and col_b:
            >>> df.rows['A','a'] += 0
            >>> df.rows['A','a'] += 1
        Inspect the index at ['A','a']:
            >>> df.rows['A','a']
            slice(0, 2, None)

        """
        # keys is a list of values where each value corresponds to the next consecutive index level
        # for example, if keys == ['A','a',3], it is assumed that _DataIndex has three-level nested
        # dicts.  This function works recursively, at each recursion taking the first value and using
        # it as the key and pussing the remaining elements to the nested dict.
        if len(keys)>1:
            # Haven't reached the last key/ inner-most dict, thus call recursively the next level 
            if not keys[0] in self.data:
                self.data[keys[0]] = _DataIndex(self.assume_sorted)
            self.data[keys[0]][keys[1:]] = dest_row_idx
        else:
            # Reached the inner-most dict, decide whether to use list or custom slicer
            # and **APPEND** the value accordingly.  Warning: this is not true assignment,
            # this is appending operation!
            ## Check if dest_row_idx is a slice or a list - that indicates that the index is being recreated from an existing index
            if isinstance(dest_row_idx,(list,slice)):
                self.data[keys[0]] = dest_row_idx
            elif not self.assume_sorted:
                # Collect list of indices
                if isinstance(self.data[keys[0]] ,list):
                    self.data[keys[0]].append(dest_row_idx)
                else:
                    self.data[keys[0]] = [dest_row_idx]
            else:
                # Build slicer objects
                if isinstance(self.data[keys[0]] ,slice):
                    self.data[keys[0]] = slice(self.data[keys[0]].start,dest_row_idx+1) # Replaces existing slice with new one by keeping the same start but modifying the end. This works because data is sorted
                else:
                    self.data[keys[0]] = slice(dest_row_idx,dest_row_idx+1)
        return

    def __getitem__(self, keys: list[Any]) -> list[int] | slice:
        """Retrieves the slicer or list of indices using keys"""
        if len(keys)>1:
            return self.data[keys[0]][keys[1:]]
        else:
            return self.data[keys[0]]

    def labels(self) -> list[Any]:
        """Lists possible labels at the current level"""
        return list(self.data.keys())

    def list_levels(self, _trail: list[Any]=[], include_slicers: bool=False) -> list[Any]:
        """Generate list of allkey combinations.
        
        All, i.e. all levels', keys are returned as a nested list.
        Each inner list contains the keys for each key column.

        Parameters
        ----------
        _trail : list
                 Used internally for recursion.

        Examples
        --------
        Create new DataFrame and set index using two columns, col_a and col_b
            >>> data = {
            >>>     'col_a': ['A','A','B','C'],
            >>>     'col_b': ['a','b','a','a'],
            >>>     'col_val': [45,46,22,91]
            >>> }
            >>> df = DataFrame(data).set_row_index(['col_a','col_b'])
        This created the "rows" attribute which is the DataFrame df', row index.
        View all possible key combinations in this index:
            >>> df.rows.list_levels()
            [['A', 'a'], ['A', 'b'], ['B', 'a'], ['C', 'a']]

        """
        resulting_levels=[]
        if isinstance(self.data[list(self.data.keys())[0]],_DataIndex):
            for key, nested_dict in self.data.items():
                this_result = nested_dict.list_levels(_trail=_trail + [key], include_slicers=include_slicers)
                resulting_levels.extend(this_result)
        else:
            if include_slicers:
                resulting_levels = [_trail + [key, slicer] for key,slicer in self.data.items()]
            else:
                resulting_levels = [_trail + [key] for key in self.data.keys()]
        return resulting_levels
    
    def as_dict(self):
        """Return description of the index sufficient to recreate it."""
        index = self.list_levels(include_slicers=True)
        index_dict = {}
        for key_and_slicer in index:
            keys = tuple(key_and_slicer[:-1])
            slicer = key_and_slicer[-1]
            index_dict[keys] = slicer
        return index_dict

# Experimental class Category, may or may not be used in the future
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

# =========================================================================
# Main Functionality
# =========================================================================

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
    fillna()
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
    def __init__(self, data: Iterable, col_properties: dict[str,Any]=None) -> None:
        """
        Initiates new column.
        
        New DataColumn containing the provided data and properties,
        if provided. If no col_properties is passed, initiates 
        all column properties with the value of None.
        

        Parameters
        ----------
        data : list
               List of the column's values.
        col_properties : dict
               Property: value pairs. Currently utilized properties
               are listed in ALLOWED_COL_PROPERTIES.
        
        """
        # Check that col_properties has been correctly specified:
        if not isinstance(col_properties, (dict,type(None))):
            raise TypeError("col_properties must be a dict or None")
        elif isinstance(col_properties,dict):
            for property_name in col_properties.keys():
                if property_name not in ALLOWED_COL_PROPERTIES:
                    raise ValueError(f"Requested to set property {property_name} which is not an allowed DataColumn property.")
        # Check that data has been correctly speficied:
        if not isinstance(data, (list, tuple, DataColumn)):
            raise TypeError("Data must be either a list or a DataColumn")
        # Proceed to defining DataColumn
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        elif isinstance(data, DataColumn):
            self.data = data.data
        # Initiate empty properites
        for property_name in ALLOWED_COL_PROPERTIES:
            setattr(self, property_name, None)
        # Set the passed properties
        if isinstance(col_properties,dict):
            # Set "straight-forward" properties
            for property_name, property_value in col_properties.items():
                if property_name in COMPLEX_COL_PROPERTIES:
                    continue
                setattr(self, property_name, property_value)
            # Custom processing for aggregation_func:
            if 'aggregation_func' in col_properties:
                # Accepts a method definition or a string.  If string, lookup corresponding definition
                if isinstance(col_properties['aggregation_func'], str):
                    setattr(self, 'aggregation_func', _agg_functions[col_properties['aggregation_func']])
                else:
                    setattr(self, 'aggregation_func', col_properties['aggregation_func'])
            # Custom processing for dtype:
            if 'dtype' in col_properties:
                if col_properties['dtype'] is not None:
                    self.data = _cast_list(self.data, col_properties['dtype'])
                    setattr(self,'dtype',col_properties['dtype'])
                
        # Additional properties:
        ## Override col_print_length.  This is done here because __init__ may be called in result of 
        ## manipulation of existing DataColumn, e.g. due to adding two columns.  In that case,
        ## whatever col_print_length was provided is not applicable any more.
        setattr(self, 'col_print_length', _calc_col_print_length(self.data,self.short_name))
        ## length
        setattr(self, 'length', len(self.data))
        return
            
    def set_properties(self, col_properties: dict[str, Any]) -> Self:
        """Returns new column with specified properties"""
        return DataColumn(self.data, col_properties)

    def _get_property(self, property_name: str) -> Any:
        """Extracts a property value"""
        try:
            return getattr(self,property_name,None)
        except:
            raise ValueError(f"Property {property_name} not found")

    def _get_all_properties(self) -> dict[str,Any]:
        """
        Extracts all properties
        
        Forms a dict of dicts that can be used to recreate this column i.e. in DataColumn
        class initialization.
        
        """
        all_properties = {}
        for prop in ALLOWED_COL_PROPERTIES:
            all_properties[prop] = self._get_property(prop)
        return all_properties

    def __getitem__(self, key: int) -> list[Any]:
        return self.data[key]

    def __len__(self) -> int:
        return self.length

    def __add__(self, other: Self | int | float | Iterable) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            col_properties = self._get_all_properties()
            col_properties['dtype'] = float if isinstance(other, float) else self._get_property('dtype')
            return DataColumn([x + other for x in self.data], col_properties=col_properties)
        elif _is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            col_properties = self._get_all_properties()
            return DataColumn([x + y for x, y in zip(iter(self),iter(other))],col_properties=col_properties)
        else:
            raise TypeError("Operands must be iterable or 'Int,' or 'Float'")

    def __sub__(self, other: Self | int | float | Iterable) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            col_properties = self._get_all_properties()
            col_properties['dtype'] = float if isinstance(other, float) else self._get_property('dtype')
            return DataColumn([x - other for x in self.data], col_properties=col_properties)
        elif _is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            col_properties = self._get_all_properties()
            return DataColumn([x - y for x, y in zip(iter(self),iter(other))],col_properties=col_properties)
        else:
            raise TypeError("Operands must be iterable or 'Int,' or 'Float'")

    def __mul__(self, other: Self | int | float | Iterable) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            col_properties = self._get_all_properties()
            col_properties['dtype'] = float if isinstance(other, float) else self._get_property('dtype')
            return DataColumn([x * other for x in self.data], col_properties=col_properties)
        elif _is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            col_properties = self._get_all_properties()
            return DataColumn([x * y for x, y in zip(iter(self),iter(other))],col_properties=col_properties)
        else:
            raise TypeError("Operands must be iterable or 'Int,' or 'Float'")

    def __truediv__(self, other: Self | int | float | Iterable) -> Self:
        if isinstance(other, int) or isinstance(other, float):
            if other==0:
                raise ValueError("Div by zero is not allowed")
            col_properties = self._get_all_properties()
            col_properties['dtype'] = float if isinstance(other, float) else self._get_property('dtype')
            return DataColumn([x / other for x in self.data], col_properties=col_properties)
        elif _is_iterable(other):
            if len(self) != len(other):
                raise ValueError("Columns have incompatible lengths")
            if 0 in other:
                raise ValueError("Encountered division by zero")
            col_properties = self._get_all_properties()
            return DataColumn([x / y for x, y in zip(iter(self),iter(other))],col_properties=col_properties)
        else:
            raise TypeError("Can only divide by the types 'Int,' or 'Float'")

    def __eq__(self, other: Self | int | float | Iterable) -> Self:
        return _element_wise_comparison(operator.eq,self, other)

    def __lt__(self, other: Self | int | float | Iterable) -> Self:
        return _element_wise_comparison(operator.lt,self, other)

    def __le__(self, other: Self | int | float | Iterable) -> Self:
        return _element_wise_comparison(operator.le,self, other)

    def __ne__(self, other: Self | int | float | Iterable) -> Self:
        
        return _element_wise_comparison(operator.ne,self, other)

    def __ge__(self, other: Self | int | float | Iterable) -> Self:
        
        return _element_wise_comparison(operator.ge,self, other)

    def __gt__(self, other: Self | int | float | Iterable) -> Self:
        
        return _element_wise_comparison(operator.gt,self, other)
        
    def __repr__(self) -> str:
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

    def as_list(self) -> list[Any]:
        """Return this column's values as a list"""
        return list(self.data)

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def apply(self, func: Callable) -> Self:
        """Map func onto this column's values"""
        return DataColumn(list(map(func,self.data)), col_properties=self._get_all_properties())

    def sum(self) -> int | float:
        """Return the sum of this column's values"""
        return sum(self.data)

    def min(self) -> int | float:
        """Return the smallest of this column's values"""
        return min(self.data)

    def max(self) -> int | float:
        """Return the largest of this column's values"""
        return max(self.data)

    def mean(self) -> int | float:
        """Return the mean of this column's values"""
        return statistics.mean(self.data)

    def median(self) -> int | float:
        """Return the median of this column's values"""
        return statistics.median(self.data)

    def median_low(self) -> int | float:
        """Return the low median of this column's values"""
        return statistics.median_low(self.data)

    def median_high(self) -> int | float:
        """Return the high median of this column's values"""
        return statistics.median_high(self.data)

    def mode(self) -> int | float:
        """Return the mode of this column's values"""
        return statistics.mode(self.data)

    def std(self) -> float:
        """Return the sample standard deviation of this column's values"""
        return statistics.stdev(self.data)

    def var(self) -> float:
        """Return the sample variance of this column's values"""
        return statistics.variance(self.data)

    def pstd(self) ->  float:
        """Return the population standard deviation of this column's values"""
        return statistics.pstdev(self.data)

    def pvariance(self) ->  float:
        """Return the population variance of this column's values"""
        return statistics.pvariance(self.data)

    def cov(self,other: Self) -> float:
        """Return the covariance of this column with other column"""
        if isinstance(other, DataColumn):
            return statistics.covariance(self.data,other.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def cor(self,other: Self) -> float:
        """Return the correlation of this column with other column"""
        if isinstance(other, DataColumn):
            return statistics.correlation(self.data,other.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def lr(self,other: Self) -> tuple[float,float]:
        """Linear regression against another column.

        Regress this column on another column and return slope and intercept.
        https://docs.python.org/3/library/statistics.html

        Returns slope, intercept
        """
        if isinstance(other, DataColumn):
            return statistics.linear_regression(other.data,self.data)
        else:
            raise TypeError("Can only compare to another DataColumn")

    def as_type(self, new_type: Any) -> Self:
        """Returns DataColumn equivalent to this but with values cast to new_type"""
        casted_values = []
        for val in self.data:
            try:
                if val==None:
                    casted_val=None
                elif isinstance(val, str) and issubclass(new_type, Number):
                    casted_val = new_type(val.replace(",",""))
                else:
                    casted_val = new_type(val)
                casted_values.append(casted_val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannon cast {val} to {new_type}: {e}")
        col_properties = self._get_all_properties()
        col_properties['dtype'] = new_type
        return DataColumn(casted_values,col_properties=col_properties)

    def isna(self) -> list[bool]:
        """Return list of bools indicating missing values"""
        return list(map(lambda x: x==None,self.data))

    def any_na(self) -> bool:
        """Return True if any value is None"""
        return any(map(lambda x: x == None,self.data))

    def fillna(self,fill_val: Any) -> Self:
        """Return DataColumn with fill_value in place of missing values"""
        new_values = list(map(lambda x: fill_val if x==None else x,self.data))
        col_properties = self._get_all_properties()
        return DataColumn(new_values,col_properties=col_properties)

    def unique(self) -> list[Any]:
        """Return list of all distinct values."""
        return(list(set(self.data)))
        
class DataFrame:
    '''
    Simplistic DataFrame

    Consists of columns represented by DataColumn class.

    Attributes
    ----------
    rows : _DataIndex
           DataIndex object used for indexing this frame's data.
    row_index_labels : list of str
                       Ordered list of labels of columns which were
                       used to build the rows index.
    _data
    columns : dict
              Label - index pairs for columns in this data frame.

    Functions
    ---------
    read_csv
    to_csv
    apply

    '''
    def __init__(self,data: dict[str,Iterable]=None,col_properties: dict[str,dict]=None,row_index: _DataIndex=None,row_index_labels: list[str]=None) -> None:
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
        col_properties : dict {str : dict}
            For each column, provided a dict of column
            properties. Defaults to None.
        row_index : dict | _DataIndex
            Used to recreate the index from a dict, or to assign
            an existing index. Warning: exercise caution when
            passing an existing index as the passed _DataIndex
            will not be re-created but referenced instead.
        row_index_labels : list
            List of labels, ordered, that were used to create 
            the row_index.
        
        """
        col_properties_provided = isinstance(col_properties,dict)
        values_len = -1
        # Initiate values for DataFrame attributes
        self.rows = _DataIndex(assume_sorted=True)
        self._data = []
        self.columns = {} # keys are short names; col_properties includes long_name
        self.row_index_labels = []
        if data==None:
            pass
        # If data was passed, build columns and append to _data
        elif isinstance(data,dict):
            col_idx = 0 # iterate column index
            for key, values in data.items():
                # Check column lengths compatibiilty
                if values_len == -1:
                    values_len = len(values)
                else:
                    if len(values) != values_len:
                        raise ValueError("Columns have incompatible lengths")
                # Check if column properties were given and store data values
                if col_properties_provided:
                    self._data.append(DataColumn(values,col_properties[key]))
                else:
                    self._data.append(DataColumn(values))
                # Add column to columns dict
                self.columns[key] = col_idx
                col_idx += 1
            if isinstance(row_index,_DataIndex):
                # Use the provided index (assumes programmer knows what they are doing here as this creates reference to whatever index was passed...)
                self.rows = row_index
            elif isinstance(row_index,dict):
                # Recreates the index based on the provided dict
                self.rows = _DataIndex(assume_sorted=True,index_desc=row_index)
            if isinstance(row_index_labels,list):
                self.row_index_labels = row_index_labels
        else:
            raise TypeError("Data must be of the type'Dict'")
        return
            
    def read_csv(self, file_path: str) -> None: ###### NEEDS TO BE UPDATED.  If data already exists, should keep column properties but update data
        """
        Read data from a csv file.

        THIS METHOD IS INCOMPLETE (It works for empty frame only but must be allowed to work with populated frames)
        
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
        return

    def to_csv(self, file_path: str) -> None:
        """
        Store this frame's data into a csv file
        
        INCOMPLETE, need to fix _data and address the questions in the comments.
        
        """
        with open(file_path, 'w', newline='') as file: # newline????
            csv_writer = csv.writer(file) # https://docs.python.org/3/library/csv.html
            csv_writer.writerow(list(self.columns.keys()))
            csv_writer.writerows(self._data)

    def __getitem__(self, key: str | list[str] | list[slice | list,str | list] | list[bool]) -> Self | DataColumn:
        """
        Select elements from the DataFrame

        Returns
        -------
        DataFrame, if multiple columns are selected.
        DataColumn, if ony column is selected

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
        Select column col_a:
            >>> df['col_a']
        To select multiple columns, use a list.  Do not use a tuple.
            >>> df[['col_a','col_val']]

        Column and row selectors:
        First 10 rows and columns col_a, col_b:
            >>> df[:10, ['col_a','col_b']]
        Rows 1 and 4, colum col_a
            >>> df[[1,4], ['col_a','col_b']]
        Rows 0 and 3, all columns:
        (recognizes that the list is of the same length as len(self))
            >>> df[[True,False,False,True]]
        Rows 1 and 4, columns col_a (by label) and 2 (by location):
            >>> df[[1,4], ['col_a',2]]
        
        """
        all_cols_properties = {}
        # Filter Rows only
        # ---------------
        if isinstance(key,list) and len(key)==len(self):
            new_data_dict = {} # Store the selected data, then use it to create and return new DataFrame
            new_cols = list(self.columns.keys()) # All columns
            # Extract row selector
            if isinstance(key,DataColumn):
                row_selector = key.data
            else:
                row_selector = key
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
            # Create new df and re-index it
            resulting_df = DataFrame(new_data_dict,col_properties=all_cols_properties).set_row_index(self.row_index_labels)
            return resulting_df
        # Filter Rows and Columns
        # ----------------------
        elif isinstance(key,tuple):
            new_data_dict = {} # Store the selected data, then use it to create and return new DataFrame
            new_cols = []
            # Extract row selector
            if isinstance(key[0],DataColumn):
                row_selector = key[0].data
            else:
                row_selector = key[0]
            # Extract column selector
            try:
                key[1]
            except:
                raise ValueError("No column selector was provided. Delete comma?")
            if isinstance(key[1],(list,tuple)):
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
            ## Loop over columns, then select appropriate rows from each column
            for col_label, col_idx in self.columns.items():
                if col_label in new_cols:
                    # Filter rows:
                    if isinstance(row_selector,list):
                        if isinstance(row_selector[0],bool):
                            new_data_dict[col_label] = [x for x, is_selected in zip(self._data[col_idx],row_selector) if is_selected]
                        elif isinstance(row_selector[0],int):
                            new_data_dict[col_label] = [self._data[col_idx][x] for x in row_selector]
                    else:
                        new_data_dict[col_label] = self._data[col_idx][row_selector]
                    all_cols_properties[col_label] = self._data[col_idx]._get_all_properties()
            # Create new df and re-index it if any indices were kept:
            resulting_df = DataFrame(new_data_dict,col_properties=all_cols_properties)
            # If self had row_index and if any key columns are retained, reindex the resulting df:
            if self.row_index_labels is not None:
                new_row_index_labels = [col for col in self.row_index_labels if col in new_data_dict.keys()]
                if len(new_row_index_labels)>0:
                    resulting_df = resulting_df.set_row_index(new_row_index_labels)
            return resulting_df
        # Filter only one Column by location
        # ----------------------------------
        elif isinstance(key, int):
            return DataColumn(self._data[key],col_properties=self._data[key]._get_all_properties())
        # Filter only one Column by label
        # -------------------------------
        elif isinstance(key, str):
            try:
                col_idx = self.columns[key]
                return DataColumn(self._data[col_idx],col_properties=self._data[col_idx]._get_all_properties())
            except ValueError:
                raise KeyError(f"Column '{key}' not found")
        # Filter Columns only
        # -------------------
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
            # Create new df and re-index it if any indices were kept:
            resulting_df = DataFrame(new_data_dict,col_properties=all_cols_properties)
            # If self had row_index and if any key columns are retained, reindex the resulting df:
            if self.row_index_labels is not None:
                new_row_index_labels = [col for col in self.row_index_labels if col in new_data_dict.keys()]
                if len(new_row_index_labels)>0:
                    resulting_df = resulting_df.set_row_index(new_row_index_labels)
            return resulting_df

    def __setitem__(self, key: str | int, new_column: DataColumn | list[Any] | Any) -> None: ## will need to delete _update_col_lengths, make sure it creates new instance of column
        """
        Modify existing column or create new column.
        
        New values must be either DataColumn, list, int, float, str, datetime, or bool.
        Setting dtype:
        if DataColumn was provided, use the same dtype.  Otherwise base on the first value of new_column
        
        """
        required_col_len = len(self._data[0])
        # Identify destination column
        if isinstance(key, str):
            col_label = key
            # If exists, find the column index, otherwise check if possible (corrent length) to create the column
            if key in self.columns:
                # Column exists
                col_idx = self.columns[key]
            else:
                # Creating new column
                col_idx = len(self.columns) # b/c current length is 1 greater than current rightmost idn
                self.columns[key] = len(self.columns)
                self._data.append(DataColumn([None]*required_col_len))
        elif isinstance(key, int):
            col_idx = key
            col_label = list(self.columns.keys())[col_idx]
        else:
            raise TypeError("Key must be of the types 'Str' or 'Int'")
        # Create new column and set in _data
        ## DataColumn was provided
        if isinstance(new_column, DataColumn):
            if required_col_len != len(new_column):
                raise ValueError("Columns have incompatible lengths")
            new_data = new_column.data
            col_properties = new_column._get_all_properties()
            self._data[col_idx] = DataColumn(new_data, col_properties=col_properties)
        ## List was provided
        elif isinstance(new_column, list):
            if required_col_len != len(new_column):
                raise ValueError("Columns have incompatible lengths")
            col_properties = self._data[col_idx]._get_all_properties()
            col_properties['dtype'] = type(new_column[0])
            self._data[col_idx] = DataColumn(new_column,col_properties=col_properties)
        ## Scalar was provided
        elif isinstance(new_column,(int,str,float,bool,datetime.datetime,Category)):
            col_props = self._data[col_idx]._get_all_properties()
            col_props['dtype'] = type(new_column)
            self._data[col_idx] = DataColumn([new_column]*required_col_len,col_properties=col_props)
        ## Unacceptable type was provided
        else:
            raise TypeError("New column values must be a list, DataColumn, Str, Int, Bool, or Datetime.")
        return
        
    def __len__(self) -> int:
        return len(self._data[0])
    
    def __repr__(self) -> str:
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

    def __iter__(self) -> Iterable:
        return iter(self._data)

    def show(self,rows: int | tuple[int,int]=5,show_index=True) -> str:
        """Print the requested rows of data.

        Parameters
        ----------
        rows : int or list or tuple
               If int, indicates how many top rows to print. 
               If list, must have format [first_row, n_rows].
        show_index : bool
                     Whether to print index.
        """
        if isinstance(rows,int):
            start_row=0
            nrows=rows
        elif isinstance(rows, (list,tuple)):
            start_row=rows[0]
            nrows=rows[1]
        screen_max_x = 80
        # Calculate how many columns we are able to display on the screen, assuming screen size screen_max_x
        # Header rows will keep track of columns in their respective loops
        # Data rows will be limited by limiting the display_data nested list's dimensions
        prefix_extra_len = len(str(start_row+nrows))-1
        screen_cols_length = prefix_extra_len
        screen_cols_num = 0
        while screen_cols_length < screen_max_x and screen_cols_num < len(self.columns):
            screen_cols_length += self._data[screen_cols_num].col_print_length
            screen_cols_num += 1
        screen_cols_num -= 1
        # Prep
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
        for col_idx, col in enumerate(self._data):
            if col_idx > screen_cols_num:
                break
            col = list(it.islice(col,start_row,start_row+nrows))
            display_data.append(col)
        # Transpose for  printing row by row
        display_data = list(zip(*display_data))
        # Print header
        ## Row 1 (short name)
        row_1_string = ""
        row_1_string += prefix_header1 + " "
        for col_label, col_idx in self.columns.items():
            if col_idx > screen_cols_num:
                break
            col_width = self._data[col_idx].col_print_length
            row_1_string += f"{col_label[:self._data[col_idx].col_print_length]:^{col_width}}" + ' | '
        print(row_1_string)
        ## Row 2 (dtypes)
        print(prefix_header2,end=' ')
        for col_label, col_idx in self.columns.items():
            if col_idx > screen_cols_num:
                break
            try:
                dtype = self._data[col_idx].dtype
                col_width = self._data[col_idx].col_print_length
                text_to_print = ""
                if dtype==str:
                    text_to_print = _pretty_string(f"{'str':>{col_width}}",'magenta')
                elif dtype==int or dtype==float:
                    text_to_print = _pretty_string(f"{'num':>{col_width}}",'green')
                elif dtype==Category:
                    text_to_print = _pretty_string(f"{'C':>{col_width}}",'yellow') ########################### Need to specify whether dummiefied already or not and how many cats
                else:
                    text_to_print = _pretty_string(f"{'UNK':>{col_width}}",'red')
                print(text_to_print,end = ' | ')
            except:
                pass
        ## Row 3 (aggregation summary)
        print()
        print(prefix_header3,end=' ')
        for col_label, col_idx in self.columns.items():
            if col_idx > screen_cols_num:
                break
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
            
            text_to_print = f"{agg_funct_string:>{col_width-len(warning_string)}}"+_pretty_string(warning_string,'red')
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
                    text_to_print = _pretty_string(f"{'--':>{col_width}}",'red')
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

    def set_property(self,property_type: str,new_properties: dict[str,Any]) -> Self:
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
        new_data = self.as_dict()
        new_col_properties = {}
        for col_name, col_idx in self.columns.items():
            new_col_properties[col_name] = self._data[col_idx]._get_all_properties()
            if col_name in new_properties.keys():
                new_col_properties[col_name][property_type] = new_properties[col_name] # bc returns new column
        # get row index
        if len(self.row_index_labels)>0:
            row_index_labels = self.row_index_labels
            row_index_dict = self.rows.as_dict()
        else:
            row_index_labels = None
            row_index_dict = None
        return DataFrame(new_data,col_properties=new_col_properties,row_index=row_index_dict,row_index_labels=row_index_labels)

    def set_short_col_names(self,new_names: dict[str,str],promote_current_to_long_names: bool=False) -> Self:
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
        # Make sure dict was passed
        if not isinstance(new_names,dict):
            raise TypeError("new_names must be a dict")
        # Make sure none of the new names is not already taken
        for new_label in new_names.values():
            if new_label in self.columns.keys():
                raise ValueError(f"Column {new_label} already exists")
        # Build a dicts for data and column properties. Iterate to keep current col order while updating the names
        new_data = {}
        new_col_props = {}
        new_row_index_labels = None
        new_row_index = None
        for cur_label, col_idx in self.columns.items():
            if cur_label in new_names:
                new_label = new_names[cur_label]
                new_data[new_label] = self._data[col_idx].data
                new_col_props[new_label] = self._data[col_idx]._get_all_properties()
                new_col_props[new_label]['short_name'] = new_label
                new_col_props[new_label]['long_name'] = cur_label if promote_current_to_long_names else None
            else:
                new_data[cur_label] = self._data[col_idx].data
                new_col_props[cur_label] = self._data[col_idx]._get_all_properties()
        # get row index
        if len(self.row_index_labels)>0:
            new_row_index_labels = self.row_index_labels
            row_index_dict = self.rows.as_dict()
        else:
            row_index_labels = None
            row_index_dict = None
        return DataFrame(new_data, col_properties=new_col_props, row_index=row_index_dict, row_index_labels=new_row_index_labels)

    def get_col_names(self) -> dict[str,int]:
        """Get dict of column names  (short : long)"""
        col_names_dict = {}
        for col_short_name, col_idx in self.columns.items():
            try:
                col_names_dict[col_short_name] = self.col_properties[col_idx].long_name
            except AttributeError:
                col_names_dict[col_short_name] = None
        return col_names_dict

    def set_row_index(self,key_col_labels: list[str],return_index: bool=False) -> Self | _DataIndex:
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

        rows = _DataIndex(assume_sorted=True)
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
            # Potential efficiency increase -- should df __init__ take _DataIndex as argument in addition to a dict?
            resulting_data_frame = DataFrame(new_data,col_properties=new_col_properties,row_index=rows,row_index_labels=key_col_labels)
            return resulting_data_frame
        else:
            return rows

    def aggregate(self,ignore_na: bool=False) -> Self:
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
        new_data_dict = {}
        new_rows_index = _DataIndex(assume_sorted=True,index_desc=self.rows.as_dict())
        new_rows_index_labels = list(self.row_index_labels)
        new_col_props ={}
        for col_label, col_idx in self.columns.items():
            new_data_dict[col_label] = []
            new_col_props[col_label] = self._data[col_idx]._get_all_properties()
        # Begin aggregation process
        for row_idx, augmented_keys in enumerate(self.rows.list_levels(include_slicers=True)):
            slicer = augmented_keys[-1]
            keys = augmented_keys[:-1]
            # Aggregate within this key combination/key group
            for col_label, col_idx in self.columns.items():
                # Handle key columns
                if self._data[col_idx].key:
                    new_value = self._data[col_idx][slicer][0]
                    new_data_dict[col_label].append(new_value)
                # Handle non-key i.e. measure/aggregate columns
                else:
                    values = self._data[col_idx][slicer]
                    if ignore_na:
                        values = [x for x in values if x is not None]
                    if not values:
                        new_value = None
                    else:
                        agg_func = self._data[col_idx].aggregation_func
                        new_value = agg_func(values)
                    new_data_dict[col_label].append(new_value)
            new_rows_index[keys] = row_idx
        return DataFrame(new_data_dict, new_col_props, new_rows_index, new_rows_index_labels)

       #new_data = [[] for col in self.columns.items()]
       #row_idx = 0
       ## Iterate over all unique key combinations
       #for key_values in self.rows.list_levels():
       #    # Aggregate values within this key combination, one column at a time
       #    for col_name, col_idx in self.columns.items():
       #        # Columns marked as keys are not aggregated -- they are the keys
       #        if self._data[col_idx].key:
       #            current_value = self._data[col_idx][self.rows[key_values]][0]
       #            new_data[col_idx].append(current_value)
       #        else:
       #            # Apply this column's aggregation function
       #            current_values = self._data[col_idx][self.rows[key_values]]
       #            if ignore_na:
       #                current_values = [value for value in current_values if value is not None]
       #            if not current_values:
       #                new_value = None
       #            else:
       #                new_value = self._data[col_idx].aggregation_func(current_values)
       #            new_data[col_idx].append(new_value)
       #    self.rows[key_values] = row_idx
       #    row_idx += 1
       ## Recreate dataframe one column at a time
       #for col_label, col_idx in self.columns.items():
       #    new_col_data = new_data[col_idx]
       #    new_col_props = self._data[col_idx]._get_all_properties()
       #    self._data[col_idx] = DataColumn(new_col_data,col_properties=new_col_props)
       #return

    def values(self, transpose: bool=False, skip_col_labels: list[str]=[], return_labels: bool=False) -> list[Any] | tuple[list[Any],list[str]]:
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
        
    def as_dict(self) -> dict[str,list[Any]]:
        """
        Returns a dict that represents this data frame
        """
        data_dict = {}
        for col_label, col_idx in self.columns.items():
            data_dict[col_label] = self._data[col_idx].data
        return data_dict

    def join(self, other: Self) -> Self:
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
        # Helper:
        n_left_cols = len(l_col_labels) # remember # of rows in left df to start right cols' indices correctly
        # Iterate over all index keys and construct new data
        ## This is done row at a time since data was tranposed
        for keys in l_index.list_levels():
            l_slicer = l_index[*keys]
            l_data = l_full_data[l_slicer]
            #l_len = len(l_data)
            r_slicer = r_index[*keys]
            if r_slicer == None:
                # Create dummy r_data while keeping correct number of columns
                r_data = [[None] for _ in r_col_labels]
                #r_len = 1
            else:
                r_data = r_full_data[r_slicer]
                #r_len = len(r_data)
            # cross product of left and right rows (it.product()). However, for each combination,
            # product() returns a two-tuple of the lists, so I use it.chain to flatten them
            joined_data = [list(it.chain(x[0],x[1])) for x in list(it.product(l_data,r_data))]
            new_data.extend(joined_data)
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
        resulting_df = DataFrame(new_data_dict, col_properties=new_data_props).set_row_index(
            key_col_labels=self.row_index_labels
        )
        return resulting_df

