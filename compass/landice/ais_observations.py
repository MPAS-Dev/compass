# Antarctic data from:
# Rignot, E., Bamber, J., van den Broeke, M. et al. Recent Antarctic ice mass
# loss from radar interferometry and regional climate modelling. Nature Geosci
# 1, 106-110 (2008). https://doi.org/10.1038/ngeo102
# Table 1: Mass balance of Antarctica in gigatonnes (10^12 kg) per year by
# sector for the year 2000
# https://www.nature.com/articles/ngeo102/tables/1
# and
# Rignot, E., S. Jacobs, J. Mouginot, and B. Scheuchl. 2013. Ice-Shelf Melting
# Around Antarctica. Science 341 (6143): 266-70.
# https://doi.org/10.1126/science.1235798.

# Note: May want to switch to input+, net+
# Note: Some ISMIP6 basins combine multiple Rignot basins.
#       May want to separate if we update our regions.

# key = ISMIP6 basin name
# name = informal name for ease of reference
# input = grounded SMB [estimate, uncertainty]
# outflow = grounding line flux [estimate, uncertainty]
# net = difference between input and outflow [estimate, uncertainty]
# shelf_melt = ice-shelf basal melt flux [estimate, uncertainty]
#     Note: shelf_melt uncertainty needs to be added yet to most basins
# shelfArea = ice-shelf area (km2)
# All values are Gt/yr unless otherwise noted
# Note: some basins are combinations of two Rignot basins

ais_basin_info = {
    'ISMIP6BasinAAp': {
        'name': 'Dronning Maud Land',
        'input': [60, 9],
        'outflow': [60, 7],
        'net': [0, 11],
        'shelf_melt': [57.5]},
    'ISMIP6BasinApB': {
        'name': 'Enderby Land',
        'input': [39, 5],
        'outflow': [40, 2],
        'net': [-1, 5],
        'shelf_melt': [24.6]},
    'ISMIP6BasinBC': {
        'name': 'Amery-Lambert',
        'input': [73, 10],
        'outflow': [77, 4],
        'net': [-4, 11],
        'shelf_melt': [35.5, 23.0]},
    'ISMIP6BasinCCp': {
        'name': 'Phillipi, Denman',
        'input': [81, 13],
        'outflow': [87, 7],
        'net': [-7, 15],
        'shelf_melt': [107.9]},
    'ISMIP6BasinCpD': {
        'name': 'Totten',
        'input': [198, 37],
        'outflow': [207, 13],
        'net': [-8, 39],
        'shelf_melt': [102.3]},
    'ISMIP6BasinDDp': {
        'name': 'Mertz',
        'input': [93, 14],
        'outflow': [94, 6],
        'net': [-2, 16],
        'shelf_melt': [22.8]},
    'ISMIP6BasinDpE': {
        'name': 'Victoria Land',
        'input': [20, 1],
        'outflow': [22, 3],
        'net': [-2, 4],
        'shelf_melt': [22.9]},
    'ISMIP6BasinEF': {
        'name': 'Ross',
        'input': [61 + 110, (10**2 + 7**2)**0.5],
        'outflow': [49 + 80, (4**2 + 2**2)**0.5],
        'net': [11 + 31, (11**2 + 7**2)**0.5],
        'shelf_melt': [70.3]},
    'ISMIP6BasinFG': {
        'name': 'Getz',
        'input': [108, 28],
        'outflow': [128, 18],
        'net': [-19, 33],
        'shelf_melt': [152.9]},
    'ISMIP6BasinGH': {
        'name': 'Thwaites/PIG',
        'input': [177, 25],
        'outflow': [237, 4],
        'net': [-61, 26],
        'shelf_melt': [290.9]},
    'ISMIP6BasinHHp': {
        'name': 'Bellingshausen',
        'input': [51, 16],
        'outflow': [86, 10],
        'net': [-35, 19],
        'shelf_melt': [76.3]},
    'ISMIP6BasinHpI': {
        'name': 'George VI',
        'input': [71, 21],
        'outflow': [78, 7],
        'net': [-7, 23],
        'shelf_melt': [152.3]},
    'ISMIP6BasinIIpp': {
        'name': 'Larsen A-C',
        'input': [15, 5],
        'outflow': [20, 3],
        'net': [-5, 6],
        'shelf_melt': [32.9]},
    'ISMIP6BasinIppJ': {
        'name': 'Larsen E',
        'input': [8, 4],
        'outflow': [9, 2],
        'net': [-1, 4],
        'shelf_melt': [4.3]},
    'ISMIP6BasinJK': {
        'name': 'FRIS',
        'input': [93 + 142, (8**2 + 11**2)**0.5],
        'outflow': [75 + 145, (4**2 + 7**2)**0.5],
        'net': [18 - 4, (9**2 + 13**2)**0.5],
        'shelf_melt': [155.4]},
    'ISMIP6BasinKA': {
        'name': 'Brunt-Stancomb',
        'input': [42 + 26, (8**2 + 7**2)**0.5],
        'outflow': [45 + 28, (4**2 + 2**2)**0.5],
        'net': [-3 - 1, (9**2 + 8**2)**0.5],
        'shelf_melt': [10.4]},
    'Thwaites': {
        'name': 'Thwaites',
        'input': [82.4, 4.9],
        'outflow': [118.4, 3.93],
        'net': [82.4 - 118.4, (4.9**2 + 3.93**2)**0.5],
        'shelf_melt': [57.8, 114.5]}}
