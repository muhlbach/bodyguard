#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import time, random
from datetime import datetime
 
# User
from .tools import print2
#------------------------------------------------------------------------------
# Tools to start and stopping scripts
#------------------------------------------------------------------------------
def initialize_script(seed=1991):

    # Start Timer
    t0 = time.time()

    # Set seed
    random.seed(1991)

    # Get current time and date
    now = datetime.now()
    
    print2(f"This script was initialized {now.strftime('at %H:%M:%S on %B %d, %Y')}")

    return t0


def end_script(t0):

    # Stop Timer
    t1 = time.time()
    
    # Set seed
    random.seed(1991)

    # Get current time and date
    now = datetime.now()
    
    print2(f"""This script was ended {now.strftime('at %H:%M:%S on %B %d, %Y')}:

   It took:
       {int(t1-t0)} seconds,
       {int((t1-t0)/60)} minutes, and
       {round((t1-t0)/3600,2)} hours
    """)
