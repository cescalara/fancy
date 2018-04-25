from matplotlib import pyplot as plt

__all__ = ['PlotStyle', 'Solarized']

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
                self.cmap = plt.get_cmap(cmap_name)
            except:
                self.cmap = plt.get_cmap('viridis')
        else:
            self.has_vapeplot = True
            try:
                self.cmap = vapeplot.cmap(cmap_name)
            except:
                self.cmap = plt.get_cmap(cmap_name)
                
        try:
            plt.style.use(stylesheet_name)
        except:
            self.has_custom_styles = False
            #plt.style.use('seaborn-dark')
        else:
            self.has_custom_styles = True

        self.alpha_level = 0.5

        self.textcolor = Solarized().base01


class Solarized():
    """
    Container for solarized hex colors.
    """

    def __init__(self):
        """
        Container for solarized hex colors.
        
        Colors are stored as described here:
        http://ethanschoonover.com/solarized
        """

        # base palette
        # from dark -> light
        self.base03 = '#002b36'
        self.base02 = '#073642'
        self.base01 = '#586e75'
        self.base00 = '#657b83' 
        self.base0 = '#839496'
        self.base1 = '#93a1a1'
        self.base2 = '#eee8d5'
        self.base3 = '#fdf6e3'

        # colors
        self.yellow = '#b58900'
        self.orange = '#cb4b16'
        self.red = '#dc322f'
        self.magenta = '#d33682'
        self.violet = '#6c71c4'
        self.blue = '#268bd2'
        self.cyan = '#2aa198'
        self.green = '#859900'
        
