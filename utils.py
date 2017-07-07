__all__=['take_slice']

def take_slice(arr,sls,axis):
    '''take using slices.'''
    return arr[(slice(None),)*axis+(sls,)]
