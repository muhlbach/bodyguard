#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from plotnine.ggplot import ggplot as ggplot_type
from plotnine import (ggplot, aes, geom_line, geom_point, geom_hline, geom_vline,geom_bar,geom_tile,geom_text,
                      labs, theme_classic, theme,
                      scale_y_reverse,
                      scale_y_continuous, scale_x_continuous, scale_x_datetime, scale_color_manual, scale_linetype_manual,
                      scale_fill_gradient2, scale_fill_gradient,
                      element_text, element_blank, guide_legend, guides)
from .sanity_check import check_str, check_type
from .exceptions import WrongInputException

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def heatmap(df,
            annotated_text_size=8,
            axis_tick_text_size=14,
            legend_pad=0.25,
            plot_margin=0.1,
            aspect_ratio=0.75):    
    """
    Plot a heatmap of correlation between all columns
    Make sure to use index
    """
    check_type(x=df,allowed=pd.DataFrame,name="df")
    
    # Break link
    df = df.copy()
    
    # Fix index
    df.reset_index(drop=True,inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):    
        df.columns = [" | ".join([str(c) for c in col]).strip() for col in df.columns]
    
    # Construct correlation
    df_corr = df.corr()
    
    # Remove lower-tri
    m,n = df_corr.shape
    df_corr[:] = np.where(np.arange(m)[:,None] > np.arange(n),np.nan,df_corr)
    
    # Pivot
    df_pivot = df_corr.melt(var_name="Var2", value_name='Correlation', ignore_index=False)
    df_pivot.index.name = "Var1"
    df_pivot.reset_index(inplace=True)
    df_pivot.dropna(inplace=True)
    
    # Round correlation
    df_pivot["Correlation"] = df_pivot["Correlation"].round(2)
    
    # Find categories
    categories = df_pivot["Var1"].unique().tolist()
    
    # Convert to categorical variables
    df_pivot["Var1"] = pd.Categorical(values=df_pivot["Var1"], categories=categories)
    df_pivot["Var2"] = pd.Categorical(values=df_pivot["Var2"], categories=list(reversed(categories)))
    

    gg = (
        # Initialize plot
        ggplot(data=df_pivot,
               mapping=aes(x="Var1",
                           y="Var2",
                           fill="Correlation"))
        
        # Add tiles (add space between tiles)
        + geom_tile(aes(width=0.95, height=0.95))
           
        + scale_fill_gradient2(high="green",
                               mid="white",
                               low="red",
                               midpoint=0,
                               limits=[-1,1])
        
        # Annotate
        + geom_text(aes(label='Correlation'), size=annotated_text_size)
        
        # Axis labs
        + labs(x="",y="")
        
        # Add classic theme
        + theme_classic()
        
        + theme(
            # Legend
            legend_position="none",
            legend_title=element_blank(),
            legend_box_spacing=legend_pad,
            legend_text=element_text(size=axis_tick_text_size),

            # Axis
            axis_text_x=element_text(size=axis_tick_text_size, rotation=45, hjust=1, vjust=1),
            axis_text_y=element_text(size=axis_tick_text_size),

            # General
            text=element_text(family="Times New Roman",
                              style="normal",
                              weight="normal"),
            # plot_margin=plot_margin,
            # aspect_ratio=aspect_ratio
            )
        )
        
    return gg

def save_gg(filename, gg):
    check_type(x=gg,allowed=ggplot_type,name="gg")
    
    
    if filename.endswith("eps"):
        format_opt = "eps"
    elif filename.endswith("pdf"):
        format_opt = "pdf"
    else:
        raise WrongInputException(x=filename,
                                  allowed=["eps","pdf"],
                                  name="End of 'filename'")
    
    gg.save(filename=filename,
                   format=format_opt,
                   path=None,
                   dpi= 100,
                   limitsize=False,
                   verbose= False,
                   bbox_inches= 'tight',
                   # pad_inches= 0.5
                   )
   
    
    
    
    


    