#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class WrongInputException(Exception):
    """
    Exception to be raised when input is not as expected
    """
    def __init__(self, x, allowed, name, message="\nUser-provided argument '{0}' is currently '{1}'.\nArgument must be one of: '{2}'"):
        self.x = x
        self.allowed = allowed
        self.name = name
        self.message = message
        super().__init__(self.message)    

    def __str__(self):
        
        return self.message.format(self.name,self.x,self.allowed)


class WrongInputTypeException(Exception):
    """
    Exception to be raised when input is of the expected instance
    """
    def __init__(self, x, allowed, name, message="\nUser-provided argument '{0}' is currently an instance of '{1}'.\nArgument must be an instance of either one of: '{2}'"):
        self.x = x
        self.allowed = allowed
        self.name = name
        self.message = message
        super().__init__(self.message)    

    def __str__(self):
        
        return self.message.format(self.name,type(self.x),self.allowed)


def print_warning(msg=""):
    
    print(
f"""
------------------------- WARNING -------------------------
{msg}
-----------------------------------------------------------
"""
        )
