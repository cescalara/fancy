
class Uhecr():
    """
    Stores the data and parameters for UHECRs
    """

    
    def __init__(self, data):
        """
        Stores the data and parameters for UHECRs.
        
        :param data: data passed as an instance of Data
        """

        self.N = data.get_len()
        
        self.coord = data.get_coordinates()

        self.year = data.get_by_name('year')

        self.day = data.get_by_name('day')

        self.incidence_angle = data.get_by_name('incidence angle')

        self.energy = data.get_by_name('energy')

        
        
