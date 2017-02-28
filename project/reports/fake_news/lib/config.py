import matplotlib as mpl
from cycler import cycler

CATEGORIES = ['left', 'mainstream', 'right']
RATINGS =  ['no factual content', 'mostly false', 'mixture of true and false', 'mostly true']

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

mpl.rcParams.update({
    'axes.prop_cycle': cycler(color=palette),
    'font.sans-serif': 'Helvetica',
    'font.style': 'normal',
    'font.size': 10,
})

page_index = ['AddictingInfoOrg', 'OccupyDemocrats', 'TheOther98', 
              'ABCNewsPolitics', 'cnnpolitics', 'politico', 
              'FreedomDailyNews', 'OfficialRightWingNews', 'theEagleisRising']

categories = {
    'politico': 'mainstream', 
    'cnnpolitics': 'mainstream', 
    'ABCNewsPolitics': 'mainstream', 
    'TheOther98': 'left', 
    'AddictingInfoOrg': 'left',
    'OccupyDemocrats': 'left', 
    'theEagleisRising': 'right', 
    'OfficialRightWingNews': 'right', 
    'FreedomDailyNews': 'right'
}

page_name_map = {
    'politico': 'Politico', 
    'cnnpolitics': 'CNN Politics', 
    'ABCNewsPolitics': 'ABC News Politics', 
    'TheOther98': 'The Other 98%', 
    'AddictingInfoOrg': 'Addicting Info',
    'OccupyDemocrats': 'Occupy Democrats', 
    'theEagleisRising': 'The Eagle is Rising', 
    'OfficialRightWingNews': 'Right Wing News', 
    'FreedomDailyNews': 'Freedom Daily'
}

color_idx = {
    # Color scheme of pages
    'politico': 0, 
    'cnnpolitics': 1, 
    'ABCNewsPolitics': 2, 
    'TheOther98': 3, 
    'AddictingInfoOrg': 4,
    'OccupyDemocrats': 5, 
    'theEagleisRising': 6, 
    'OfficialRightWingNews': 7, 
    'FreedomDailyNews': 8,
    # Color scheme of categories
    'mainstream': 2,
    'left': 0,
    'right': 3
}

def get_color(category):
    return palette[color_idx[category]]