import csv

class DataFrame:
    '''
    Simplistic DataFrame

    Functions
    ---------
    read_csv
    to_csv
    apply
    '''
    def __init__(self):
        self.data = []
        self.columns = []

    def read_csv(self, file_path):
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file) # https://docs.python.org/3/library/csv.html
            self.columns = next(csv_reader)
            self.data = [row for row in csv_reader]

    def to_csv(self, file_path):
        with open(file_path, 'w', newline='') as file: # newline????
            csv_writer = csv.writer(file) # https://docs.python.org/3/library/csv.html
            csv_writer.writerow(self.columns)
            csv_writer.writerows(self.data)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[self.columns.index(key)] for row in self.data]
        elif isinstance(key, int):
            return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            col_idx = self.columns.index(key)
            for i, row in enumerate(self.data):
                row[col_idx] = value[i]
        elif isinstance(key, int):
            self.data[key] = value
        
    def __len__(self):
        return len(self.data) # Could make it more efficient by keeping track of length throughout ops
    
    def __repr__(self):
        # Human readable representation or informal, string, representation of the dataframe
        return str(self.data)

    def __iter__(self):
        return iter(self.data)

    def apply(self, func, column):
        col_idx = self.columns.index(column)
        for row in self.data:
            row[col_idx] = func(row[col_idx])


            


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