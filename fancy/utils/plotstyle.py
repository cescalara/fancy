from matplotlib import pyplot as plt

__all__ = ['PlotStyle']

class PlotStyle():
    """
    Handles the style of plots for different environments,
    using stylesheets and custom color palettes
    """

    
    def __init__(self, cmap_name = None, stylesheet_name = None):
        """
        Handles the style of plots for different environments.

        :param cmap_name: name of the desired colormap
        :param stylesheet_name: anme of the desired stylesheet

        TODO: improve stylesheets to avoid vapeplots import
        """

        # defaults
        if cmap_name == None and stylesheet_name == None:
            cmap_name = 'sunset'
            stylesheet_name = 'custom_solarized_dark'

        # try to load, if fails revert to standard
        # matplotlib options
        try:
            import vapeplot
        except:
            self.has_vapeplot = False
            try: 
                self.cmap = plt.cmap(cmap_name)
            except:
                self.cmap = plt.cmap('viridis')
        else:
            self.has_vapeplot = True
            try:
                self.cmap = vapeplot.cmap(cmap_name)
            except:
                self.cmap = plt.cmap(cmap_name)
                
        try:
            plt.style.use(stylesheet_name)
        except:
            self.has_custom_styles = False
            plt.style.use('seaborn-dark')
        else:
            self.has_custom_styles = True

        self.alpha_level = 0.5
        self.textcolor = '#93a1a1'
