roadDict = {
    276183: 0,
    276184: 1,
    275911: 2, 
    275912: 3,
    276240: 4,
    276241: 5,
    276264: 6,
    276265: 7,
    276268: 8,
    276269: 9,
    276737: 10,
    276738: 11
}

groupDict = {
    276183: 0,
    276184: 1,
    275911: 2, 
    275912: 3,
    276240: 4,
    276241: 5,
    276264: 6,
    276265: 7,
    276268: 8,
    276269: 9,
    276737: 10,
    276738: 11
}

inverseDict = {
    0: 276183,
    1: 276184,
    2: 275911,
    3: 275912,
    4: 276240,
    5: 276241,
    6: 276264,
    7: 276265,
    8: 276268,
    9: 276269,
    10: 276737,
    11: 276738
}


def iimap(road_id):
    """
    map road_id to index
    """
    return roadDict.get(road_id, -1)


def igmap(road_id):
    """
    map road_id to group_id
    """
    return groupDict.get(road_id, -1)


def iv_map(group):
    """
    map index to group
    """
    return inverseDict.get(group, -1)
