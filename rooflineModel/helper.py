import numpy as np
import pandas as pd
from functools import reduce, wraps
from collections.abc import Iterable 

def helper(ignores=[]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            parts = []
            temp = dict(kwargs)
            for key in temp:
                if key not in ignores:
                    parts.append((key, kwargs.pop(key)))
            return func(*args, parts, **kwargs)
        return wrapper
    return decorator

def innerSelect(df, parts, sort, mapper, reducer):
    reds = {
        "or": lambda x,y: x | y,
        "and": lambda x,y: x & y
    }
    maps = {
        "eq": lambda x: df[x[0]] == x[1],
        "nq": lambda x: df[x[0]] != x[1],
        "lt": lambda x: df[x[0]] < x[1],
        "gt": lambda x: df[x[0]] > x[1],
        "le": lambda x: df[x[0]] <= x[1],
        "ge": lambda x: df[x[0]] >= x[1],
    }
    ap = map(maps[mapper], parts)
    id = reduce(reds[reducer], ap)
    ret = df[id]
    if sort is not None:
        ret = ret.sort_values(by=sort)
    return ret

# The following are some helper functions
@helper(ignores=['sort','mapper','reducer'])
def select(df, parts, sort=None, mapper="eq", reducer="and"):
    return innerSelect(df, parts, sort, mapper, reducer)

def innerGenerator(df, parts, sort, verbose, labels):
    # JS: There must be something here...
    assert(len(parts))
    # JS: We will use the helper format
    assert(isinstance(parts[0][1], Iterable))
    
    if sort is not None:
        parts[0][1].sort()

    if labels is not None:
        labels.append(None)

    for p in parts[0][1]:
        newDf = innerSelect(df, [(parts[0][0], p)], None, "eq", "and")
        if len(parts) == 1:
            if verbose:
                print(parts[0][0], "=", p, len(newDf), flush=True)
            if labels is not None:
                labels[-1] = p
                yield newDf, labels
            else:
                yield newDf
        else:
            if verbose:
                print(parts[0][0], "=", p, len(newDf), end=' ')
            if labels is not None:
                labels[-1] = p
                newGen = innerGenerator(newDf, parts[1:], sort, verbose, labels[:])
            else:
                newGen = innerGenerator(newDf, parts[1:], sort, verbose, None)
            for x in newGen:
                yield x

@helper(ignores=['columns', 'verbose', 'sort', 'labels'])
def generator(df, parts, columns=None, sort=False, verbose=False, labels=False):
    if len(parts) == 0:
        if columns is None:
            columns = df.columns
        parts = [(col, df[col].unique()) for col in columns]
        sort = True
    if verbose:
        print(parts)
    if labels:
        labels = []
    else:
        labels = None
    newGen = innerGenerator(df, parts, sort, verbose, labels)
    for x in newGen:
        yield x

def columnFormat(df):
    return df.rename(columns={old:'_'.join(old.split(' ')).lower() for old in df.columns})

def getUniqueCols(df, columns=None):
    if columns is None:
        columns = df.columns
    parts = {col: df[col].unique() for col in columns}
    for col in columns:
        parts[col].sort()
    return map(parts.get, columns)

def getAveFromQuartile(df, which):
    desc = df.describe()
    mean = desc[which]["mean"]
    stdPercent = (desc[which]["std"] / mean) * 100
    lower = desc.loc["25%", which]
    upper = desc.loc["75%", which]
    temp = df[(df[which] >= lower) & (df[which] <= upper)][which]
    return temp.mean(), mean, stdPercent

def getStat(df, which):
    desc = df[which].describe()
    return desc["min"], desc["max"], desc["50%"], desc["mean"]

def getClosest(df, metric, value):
    dist = (df[metric] - value).abs()
    ret = df.loc[[dist.idxmin()]]
    return ret
