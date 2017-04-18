class Catalan(object):
    def __init__(self):
        """
        Returns catalan numbers: 1 1 2 5 14 42 132...
        
        Memoized. Should run in amortized O(1) time.

        """
        self.sofar = 0
        self.cache = [1]

    def catalan_it(self, start=0, end=0, c=1):
        for n in range(start + 1, end + 2):
            yield c
            c = c * 2 * (2*n-1) / (n+1)
            if n > self.sofar:
                self.sofar = self.sofar + 1
                self.cache.append(c)
            
    def catalan(self, n):
        if n <= self.sofar:
            return self.cache[n]
        else:
            for c in self.catalan_it(start=self.sofar, end=n, c=self.cache[self.sofar]):
                pass
            return c


class CatalanPyramid(object):
    def __init__(self):
        self.cat = Catalan()

    def build_pyramid(self, n):
        """
        Returns list of list of lists representing fractions in the Catalan Pyramid.

        """
        
        ret = [
            [(1, 2)]
            ]
        
        for i in range(1, n):
            n_i = i + 1
            row = [[None, None] for _ in range(n_i)]
            
            # Rule 1: Right Diagonal Numerator
            row[0][0] = 1
            
            # Rule 2: Right Diagonal Denominator
            row[0][1] = i + 2
            
            # Rule 3: Left Diagonal Denominator
            row[n_i-1][1] = self.cat.catalan(i+2)

            # Rule 4: Left Diagonal Numerator
            row[n_i-1][0] = row[n_i-1][1] - ret[-1][n_i-2][1]
            
            for j in range(1, n_i-1):
                # Rule 5: Rest of Numerators
                row[j][0] = row[j-1][1]
            
                # Rule 6: Rest of Denominators
                row[j][1] = row[j][0] + ret[-1][j][1]
            
            ret.append(row)
        
        return ret

    def normalize_pyramid(self, pyramid):
        ret = []
        
        for row in pyramid:
            ret.append([n/float(d) for n, d in row])
        
        return ret

    def build_lookup_table(self, pyramid):
        depth = len(pyramid) + 1
        width = depth * 2 + 1
        ret = []
        
        for i in range(depth):
            row = [0.0] * width
            
            if i > 0:
                rowp = pyramid[i-1]
                for j, col in enumerate(rowp):
                    row[-(j+i+2)] = col
            
            if i == 0:
                row[-(i+2)] = 1.0
            elif i < depth - 1:
                row[-(i+2 + len(rowp))] = 1.0
            elif i == depth - 1:
                row[0] = 1.0
                row[1] = 1.0

            ret.append(row)
            
        ret = list(reversed(ret))
        
        return ret

    def lookup_table(self, n_tokens):
        pyr = self.build_pyramid(n_tokens-2)
        pyr = self.normalize_pyramid(pyr)
        table = self.build_lookup_table(pyr)
        return table


class ShiftProbabilities(object):
    def __init__(self):
        self.cache = dict()
        self.builder = CatalanPyramid()

    def prob(self, n_reduces, i, n_tokens):
        if n_tokens not in self.cache:
            self.cache[n_tokens] = self.builder.lookup_table(n_tokens)
        table = self.cache[n_tokens]
        return table[n_reduces][i]
