#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import re

from .sanity_check import check_type
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def special_characters():
    return ['&','+','@','!','=','#','?','%','<','*','$','>','^', '-', '(', '[', '{', ')', ']', '}', '.']

def contains_number(s):
    """Check if string contains any number"""
    return any(map(str.isdigit, s))

def replace_characters(string, repl="", pattern=None, strip=True):
    
    check_type(x=string,allowed=str,name="string")
    
    if pattern is not None:
        check_type(x=pattern,allowed=list,name="pattern")
    else:
        pattern = special_characters()
        
    for i in pattern:
        string = string.replace(i, repl)

    if strip:
        string = string.strip()
        
    return string
    

def make_string_compatible(string, lower=True, repl="_"):
    check_type(x=string,allowed=str,name="string")
    
    # Replace special characters
    string = replace_characters(string=string,
                                repl=repl,
                                pattern=special_characters()+[" "],
                                strip=True)
    
    # Remove doubles
    if repl != "":
        string = string.replace(repl*2, repl)
    
    # Lower
    if lower:
        string = string.lower()
    
    return string
    
def prettify_title(s, stop_words=None):
    """
    This function prettifies str by capitalizing all words except stop words
    """
    # Stop words
    if stop_words is None:
        stop_words = ['to', 'a', 'for', 'by', 'an', 'as', 'of', 'or', 'am', 'the', 'so', 'it', 'and']
    
    # Capitalize
    s = " ".join([i.title() if i.lower() not in stop_words else i.lower() for i in s.split(" ")])
    
    return s

def nth_repl(s, old, new, nth):
    """
    Replace the nth occurance of substring in string
    """
    find = s.find(old)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop until we find the nth or we find no match
    while find != -1 and i != nth:
        # find + 1 means we start searching from after the last match
        find = s.find(old, find + 1)
        i += 1
        # If i is equal to n we found nth match so replace
    if i == nth and i <= len(s.split(old))-1:
        return s[:find] + new + s[find+len(old):]
    return s    