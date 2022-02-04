#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class WrongInputException(Exception):
    """
    Exception raised when input is not a pd.DataFrame
    """
    def __init__(self, input_name, provided_input, allowed_inputs, message="\nUser-supplied argument '{0}' is currently '{1}'.\nArgument must be one of: '{2}'"):
        self.input_name = input_name
        self.provided_input = provided_input
        self.allowed_inputs = allowed_inputs
        self.message = message
        super().__init__(self.message)    

    def __str__(self):
        
        return self.message.format(self.input_name,self.provided_input, self.allowed_inputs)


class WrongInputTypeException(Exception):
    """
    Exception raised when input is not a pd.DataFrame
    """
    def __init__(self, input_name, provided_input, allowed_inputs, message="\nUser-supplied argument '{0}' is currently '{1}'.\nArgument must be one of: '{2}'"):
        self.input_name = input_name
        self.provided_input = provided_input
        self.allowed_inputs = allowed_inputs
        self.message = message
        super().__init__(self.message)    

    def __str__(self):
        
        return self.message.format(self.input_name,type(self.provided_input), self.allowed_inputs)