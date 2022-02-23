#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
from .exceptions import WrongInputTypeException

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
# Sanity checks
def check_type(x,allowed_type,name="x"):
    
    # Check; allowed_type
    if not isinstance(allowed_type,(type,tuple)):
        raise WrongInputTypeException(input_name="allowed_type",
                                      provided_input=allowed_type,
                                      allowed_inputs=[type,tuple])
        
    # Check; allowed_type
    if not isinstance(name, str):
        raise WrongInputTypeException(input_name="name",
                                      provided_input=name,
                                      allowed_inputs=str)
        
    # Perform actual sanity check
    if not isinstance(x,allowed_type):
        raise WrongInputTypeException(input_name=name,
                                      provided_input=x,
                                      allowed_inputs=allowed_type)
        
        
        