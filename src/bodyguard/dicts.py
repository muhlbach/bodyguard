#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard  
# User
from .sanity_check import check_type
from .lists import listify
from .tools import remove_empty_elements
from .tools import isin
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def convert_values_to_list(d):
    """ Turn all values into list """
    check_type(x=d,allowed=dict,name="d")
    
    d = {k:listify(x=v) for k,v in d.items()}
    
    return d

def unlist_values(d):
    """Unlist values in dict if possible"""
    # Turn all elements into list
    d = convert_values_to_list(d=d)
    
    # Unlist if possible
    d = {key:(value if len(value)>1 else value[0]) for key,value in d.items()}
    
    return d
    
def remove_empty(d):
    """Remove empty elements"""
    check_type(x=d,allowed=dict,name="d")
    
    d_clean = remove_empty_elements(x=d)
    
    return d_clean


def remove_conditionally_invalid_keys(d, invalid_values=["deprecated"]):
    """ Remove keys from dictionary for which values contain specified values"""    
    d = {k:v for k,v in d.items() if not isin(a=invalid_values,
                                              b=v,
                                              how="any",
                                              return_element_wise=False)
         }
        
    return d
    