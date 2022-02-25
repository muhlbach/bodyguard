#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
from .exceptions import WrongInputTypeException,WrongInputException
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# Sanity checks
def check_type(x,allowed,name="x"):
    
    # Check "allowed"
    if not isinstance(allowed,(type,tuple)):
        raise WrongInputTypeException(x=allowed,
                                      allowed=[type,tuple],
                                      name="allowed")
            
    # Check "name"
    if not isinstance(name, str):
        raise WrongInputTypeException(x=name,
                                      allowed=str,
                                      name="name")
            
    # Perform actual sanity check
    if not isinstance(x,allowed):
        raise WrongInputTypeException(x=x,
                                      allowed=allowed,
                                      name=name)
            
        
        
def check_str(x,allowed,name="x"):
    
    # Check input "x"
    check_type(x=x,allowed=str)
    
    # Check "allowed"
    if not isinstance(allowed,(str,list)):
        raise WrongInputTypeException(x=allowed,
                                      allowed=[str,list],
                                      name="allowed")
                
    # Check "name"
    if not isinstance(name, str):
        raise WrongInputTypeException(x=name,
                                      allowed=str,
                                      name="name")
        
    # Perform actual sanity check
    if not any(x==a for a in allowed):
        raise WrongInputException(x=x,
                                  allowed=allowed,
                                  name=name)
                        