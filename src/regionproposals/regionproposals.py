from _regionproposals import segment, rgb2hsv

__all__ = [
    'segment',
    'rgb2hsv',
    'nms',
]

def _intersection (region1, region2):
    l1, r1, t1, b1, _ = region1
    l2, r2, t2, b2, _ = region2
    width = min(r1, r2) - max(l1, l2)
    height = min(b1, b2) - max(t1, t2)

    width = max(0, width)
    height = max(0, height)

    return width * height

def _union (region1, region2):
    l1, r1, t1, b1, _ = region1
    l2, r2, t2, b2, _ = region2
 
    width = max(r1, r2) - min(l1, l2)
    height = max(b1, b2) - min(t1, t2)

    return width * height  

def _overlap (region1, region2):
    return _intersection (region1, region2) / _union (region1, region2)

def nms (regions, threshold):
    accepted_regions = []

    for region in regions:
        okay = True
        for accepted_region in accepted_regions:
            if _overlap (region, accepted_region) > threshold:
                okay = False
                break
        if okay:
            yield region
            accepted_regions += [region]
