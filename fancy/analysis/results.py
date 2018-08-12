import corner
import h5py

__all__ = ['Results']


class Results():
    """
    Manage the output of Analysis object.
    """

    def __init__(self, filename):
        """
        Manage the output of Analysis object.
        Reads in a HDF5 file containting fit/simulation
        results for further plotting, analysis and PPC.
        """

        self.filename = filename


    def get_chain(self, list_of_keys):
        """
        Returns chain of desired parameters specified by list_of_keys.
        """

        chain = {}
        with h5py.File(self.filename, 'r') as f:
            fit_output = f['output/fit/samples']
            for key in list_of_keys:
                chain[key] = fit_output[key].value

        return chain

    def get_truths(self, list_of_keys):
        """
        For the case where the analysis was based on simulated 
        data, return input values or 'truths' for desired 
        parameters specified by list_of_keys.
        """

        truths = {}
        with h5py.File(self.filename, 'r') as f:

            try:
                sim_input = f['input/simulation']
                for key in list_of_keys:
                    truths[key] = sim_input[key].value

            except:
                print('Error: file does not contain simulation inputs.')
        return truths
    

        
        
        
