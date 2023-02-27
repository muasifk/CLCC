
import numpy as np

def get_pacing_function(total_step, total_data, a, b, pacing_f):
    '''
    Inputs:
        a:[0,large-value]   percentage of total step when reaching to the full data. Start at (a*total_step, total_data)) 
        b:[0,1]             percentatge of total data at the begining of the training. Start at (0,b*total_data))
    '''
    index_start = b*total_data
    if pacing_f == 'linear':
        rate = (total_data - index_start)/(a*total_step)
        def _linear_function(step):
            return int(rate *step + index_start)
        return _linear_function
    
    elif pacing_f == 'quad':
        rate = (total_data-index_start)/(a*total_step)**2  
        def _quad_function(step):
            return int(rate*step**2 + index_start)
        return _quad_function
    
    elif pacing_f == 'root':
        rate = (total_data-index_start)/(a*total_step)**0.5
        def _root_function(step):
            return int(rate *step**0.5 + index_start)
        return _root_function
    
    elif pacing_f == 'step':
        threshold = a*total_step
        def _step_function(step):
            return int( total_data*(step//threshold) +index_start)
        return _step_function      

    elif pacing_f == 'exp':
        c = 10
        tilde_b  = index_start
        tilde_a  = a*total_step
        rate =  (total_data-tilde_b)/(np.exp(c)-1)
        constant = c/tilde_a
        def _exp_function(step):
            if not np.isinf(np.exp(step *constant)):
                return int(rate*(np.exp(step*constant)-1) + tilde_b )
            else:
                return total_data
        return _exp_function

    elif pacing_f == 'log':
        c = 10
        tilde_b  = index_start
        tilde_a  = a*total_step
        ec = np.exp(-c)
        N_b = (total_data-tilde_b)
        def _log_function(step):
            return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
        return _log_function