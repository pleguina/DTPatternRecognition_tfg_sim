class Pattern(object):
    def __init__(self, seeds, hits):
        self.seeds = seeds
        self.hits  = hits
        self.len   = len(hits)
        self.busted = False
        self.overlap = 0
        self.overlapsw = []

    def hasseed(self, hit):
        pass
        # if hit == seed:
        #     return True
        # else:
        #     return False

    def hashit(self, hit):
        for h in self.hits:
            if h[:2] == hit:
                return True
        return False

    def recoHits(self, extra = 0, reverse = 1):
        return [[h[0],reverse*h[1]+extra] for h in self.hits]

    def genHits(self, extra = 0, reverse = 1):
        return [[h[0],reverse*h[1]+extra, h[2]] for h in self.hits]

    def isEqual(self, other):
        isEqual = True
        for h in self.hits:
            if h in other.hits: continue
            else: isEqual = False
        for h in other.hits:
            if h in self.hits: continue
            else: isEqual = False
        if isEqual: 
            self.overlap += 1
        self.overlapsw.append(isEqual)
        return isEqual

def patternSorter(p):
    layers  = [h[0] for h in p[0]]
    layers  = list(dict.fromkeys(layers))
    nLayers = len(layers)
    nHits   = len(p[0])
    return nLayers*1000 + nHits
