#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def color_palette():
    
    color_palette = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499",
                     "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
    
    return color_palette

def color_palette_pairwise():
    
    color_palette = ["#88CCEE", "#332288",
                    "#CC6677", "#882255",
                    "#79AF97FF","#117733",
                    "#DDCC77","#FFCD00FF",
                    "#E377C2FF", "#9467BDFF"
                    ]
    
    return color_palette

def gg_defaults():
    
    # Various GG parameters
    GG_LINESIZE = 1.5
    GG_POINTSIZE = 4
    
    GG_AXIS_TEXT_SIZE = 20
    GG_AXIS_TICK_TEXT_SIZE = 14
    
    GG_TITLE_SIZE = 241

    GG_PLOT_MARGIN=0.1
    GG_LEGEND_PAD = 0.25
    GG_ASPECT_RATIO = 0.75
       
    GG_SETTINGS = {
           "point_size":GG_POINTSIZE,
           "line_size":GG_LINESIZE,
           "axis_text_size":GG_AXIS_TEXT_SIZE,
           "axis_tick_text_size":GG_AXIS_TICK_TEXT_SIZE,
           "title_size":GG_TITLE_SIZE,
           
           "legend_pad":GG_LEGEND_PAD,
           "plot_margin":GG_PLOT_MARGIN,
           "aspect_ratio":GG_ASPECT_RATIO
           }
    
    return GG_SETTINGS