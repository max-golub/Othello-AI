import sys, math
look = []
statWeights = [1000, 750, 100, 75] # corners stability mobility parity

class Strategy():
  def best_strategy(self, board, player, best_move, still_running):
    pzl, token = convToNormal(board, player)
    other = 'o' if token == 'x' else 'x'
    moves = findMoves(pzl, token, other)
    best_move.value = calcBestMove(pzl, moves, token, other, best_move)


def convToNormal(pzl, token):
  if token == '@': token = 'x'
  pzl = ''.join([pzl[i:i + 8] for i in range(11, 89, 10)])
  pzl = pzl.replace('@', 'x')
  token = token.lower()
  return pzl, token

def convMove(m):
  return m + (2 * ((m // 8) + 1) - 1) + 10


def findMoves(pzl, token, other):
  moves = set()
  for i in range(len(pzl)):
    if pzl[i].lower() == token:
      moves = moves | movesAtInd(pzl, i, token, other)
  return moves

def makeLookup():
  for i in range(64):
    cur = []
    cset = []
    for l in range(i + 1, i // 8 * 8 + 8):
      cset.append(l)
    cur.append(cset)
    cset = []
    for r in range(i - 1, i // 8 * 8 - 1, -1):
      cset.append(r)
    cur.append(cset)
    cset = []
    for u in range(i - 8, -1, -8):
      cset.append(u)
    cur.append(cset)
    cset = []
    for d in range(i + 8, 64, 8):
      cset.append(d)
    cur.append(cset)
    cset = []
    cc = i % 8 + 1
    cr = i // 8 - 1
    lu = i - 7
    while cr >= 0 and cc < 8:
      cset.append(lu)
      cc += 1
      cr -= 1
      lu -= 7
    cur.append(cset)
    cset = []
    cc = i % 8 + 1
    cr = i // 8 + 1
    ld = i + 9
    while cr < 8 and cc < 8:
      cset.append(ld)
      cc += 1
      cr += 1
      ld += 9
    cur.append(cset)
    cset = []
    cc = i % 8 - 1
    cr = i // 8 + 1
    rd = i + 7
    while cr < 8 and cc >= 0:
      cset.append(rd)
      cc -= 1
      cr += 1
      rd += 7
    cur.append(cset)
    cset = []
    cc = i % 8 - 1
    cr = i // 8 - 1
    ru = i - 9
    while cr >= 0 and cc >= 0:
      cset.append(ru)
      cc -= 1
      cr -= 1
      ru -= 9
    cur.append(cset)
    look.append(cur)


def alphaBeta(pzl, token, other, lower, upper, depth, bm):
  if (pzl, token, lower, upper) in cache: return cache[(pzl, token, lower, upper)]
  if (pzl, token) in mCache:
    moves = mCache[(pzl, token)]
  else:
    moves = findMoves(pzl, token, other)
    mCache[(pzl, token)] = moves
  if not moves:
    if (pzl, other) in mCache:
      oMoves = mCache[(pzl, other)]
    else:
      oMoves = findMoves(pzl, other, token)
      mCache[(pzl, other)] = oMoves
    if not oMoves: return [pzl.count(token) - pzl.count(other)]
    else:
      best = [lower - 1]
      nm = alphaBeta(pzl, other, token, -upper, -lower, depth + 1)
      score = -nm[0]
      if score > upper: return [score]
      if score >= lower:
        best = [score] + nm[1:] + [-1]
        lower = score + 1
        cache[(pzl, token, lower, upper)] = best
      return best

  best = [lower - 1]
  for m in moves:
    nm = alphaBeta(makeMove(pzl, m, token, other), other, token, -upper, -lower, depth + 1)
    score = -nm[0]
    if score > upper: return [score]
    if score < lower: continue
    best = [score] + nm[1:] + [m]
    lower = score + 1
    if depth == 0:
      bm.value = convMove(best[-1])
  cache[(pzl, token, lower, upper)] = best
  return best


def calcBestMove(pzl, moves, token, other, bm):
  if pzl.count('.') < 14:
    absBest = alphaBeta(pzl, token, other, -64, 64, 0, bm)
    return convMove(absBest[-1])
  upper = 10000000 * len(pzl)
  best = [upper]                                              # The negative of lower
  for maxDepth in range(1, 7):                               # Iterative deepening to level 6
    for m in moves:
      ab = alphabetaMG(makeMove(pzl, m, token, other), other, token, -upper, best[0], maxDepth - 1)
      if -ab[0] < -best[0]: continue                  # ab[0] & best[0] from the enemy's pt of view
      best = ab + [m]                                                # A new personal best
      bm.value = convMove(best[-1])                            # An improved move
  return convMove(best[-1])

def alphabetaMG(pzl, token, other, lower, upper, maxDepth):
  # midgame alpha/beta: returns [minBrdEval, *[reverse move sequence]]
  if (pzl, token, lower, upper, maxDepth) in cache: return cache[(pzl, token, lower, upper, maxDepth)]
  if (pzl, token) in mCache:
    moves = mCache[(pzl, token)]
  else:
    moves = findMoves(pzl, token, other)
    mCache[(pzl, token)] = moves
  if not maxDepth:                                      # Time for a board evaluation
    if (pzl, token) in scoreCache: return [scoreCache[(pzl, token)]]
    tpl = calcBoardStats(pzl, token, other, moves)
    brdVal = sum(tpl[i] * statWeights[i] for i in range(len(tpl)))
    scoreCache[(pzl, token)] = brdVal
    return [brdVal]

  if not moves:
    if (pzl, other) in mCache:
      oMoves = mCache[(pzl, other)]
    else:
      oMoves = findMoves(pzl, other, token)
      mCache[(pzl, other)] = oMoves
    if not oMoves: return [100000 * pzl.count(token) - pzl.count(other)] # ie. the real truth (unexpected game over)
    else:
      nm = alphabetaMG(pzl, other, token, -upper, -lower, maxDepth - 1)
      best = [-nm[0]] + nm[1:] + [-1]
      cache[(pzl, token, lower, upper, maxDepth)] = best
      return best



  best = [1 - lower]
  for m in moves:
    ab = alphabetaMG(makeMove(pzl, m, token, other), other, token, -upper, -lower, maxDepth-1)
    if -ab[0] > upper: return [-ab[0]]                     # Our move is too good
    if ab[0] < best[0]:                                      # Our move is an improvement
      best = ab + [m]                                     #   Note the new best
      lower = -best[0] + 1                                   #   Note the new lower bound
  nb = [-best[0]] + best[1:]
  cache[(pzl, token, lower, upper, maxDepth)] = nb
  return nb                           # return the best that we found

def movesAtInd(pzl, i, token, other):
  cset = set()
  for dir in look[i]:
    if not dir: continue
    if pzl[dir[0]] != other: continue
    for j in range(1, len(dir)):
      if pzl[dir[j]] == token: break
      if pzl[dir[j]] == '.':
        cset.add(dir[j])
        break
  return cset

def calcBoardStats(pzl, token, other, moves):
  myCorner = oCorner = 0
  for mv in {0, 7, 56, 63}:
    if pzl[mv] == token: myCorner += 1
    if pzl[mv] == other: oCorner += 1
  c = 100 * (myCorner - oCorner)
  s = 100 * stabilityScore(pzl, token, other)
  oMoves = findMoves(pzl, other, token)
  mov = len(moves)
  oMov = len(oMoves)
  if mov > oMov:
    m = (100 * mov) / (mov + oMov)
  elif oMov > mov:
    m = -(100 * oMov) / (mov + oMov)
  else: m = 0
  hole = pzl.count('.')
  p = 100 if hole % 2 == 0 else -100
  return c, s, m, p


def stabilityScore(pzl, token, other):
  count = 0
  for i, p in enumerate(pzl):
    if p != token: continue
    a = eval2Line(evalLine(pzl, look[i][0], token, other), evalLine(pzl, look[i][1], token, other))
    if a == -1:
      count += -1
      continue
    b = eval2Line(evalLine(pzl, look[i][2], token, other), evalLine(pzl, look[i][3], token, other))
    if b == -1:
      count += -1
      continue
    c = eval2Line(evalLine(pzl, look[i][4], token, other), evalLine(pzl, look[i][6], token, other))
    if c == -1:
      count += -1
      continue
    d = eval2Line(evalLine(pzl, look[i][5], token, other), evalLine(pzl, look[i][7], token, other))
    if d == -1:
      count += -1
      continue
    if a + b + c + d == 4: count += 1
  return count


def evalLine(pzl, dir, token, other):
  for d in dir:
    if pzl[d] == token: continue
    if pzl[d] == '.': return 0
    if pzl[d] == other: return -1
  return 1

def eval2Line(s1, s2):
  if s1 == 1 or s2 == 1:
    return 1
  return -1 if s1 != s2 else 0


def makeMove(pzl, moveInd, token, other):
  np = [*pzl]
  np[moveInd] = token
  for dir in look[moveInd]:
    if not dir: continue
    if np[dir[0]] != other: continue
    for j in range(1, len(dir)):
      if np[dir[j]] == '.' or np[dir[j]] == '*': break
      if np[dir[j]] == token:
        for k in range(j):
          np[dir[k]] = token
        break
  return ''.join(np)

def printPzl(pzl):
  for x in range(8):
    for y in range(8):
      print(pzl[x * 8 + y], end='')
    print()
  print()

def lowerPzl(pzl):
  np = [*pzl]
  for i, x in enumerate(np):
    np[i] = x.lower()
  return ''.join(np)

def starPzl(pzl, moves):
  np = [*pzl]
  for m in moves:
    np[m] = '*'
  return ''.join(np)

makeLookup()
pzl = '.' * 27 + 'ox......xo' + '.' * 27
token = 'x'
other = 'o'
cache = {}
mCache = {}
scoreCache = {}
print(convToNormal('???????????........??........??........??...o@...??...@o...??........??........??........???????????', 'x'))