
class Source():
    """
    Stores the data and parameters for sources
    """

    
    def __init__(self, data):
        """
        Stores the data and parameters for sources.
        
        :param data: data passed as an instance of Data
        """

        self.N = data.get_len()
        
        self.coord = data.get_coordinates()

        self.distance = data.get_by_name('D') # in Mpc

        self.name = data.get_by_name('name')

        
