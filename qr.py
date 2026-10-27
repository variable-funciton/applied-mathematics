"""QR コードを作る(外部ライブラリなし。バイトモード、誤り訂正レベル M、型番 1〜10)

アンケートの URL をホーム画面に QR コードで出すために使う。最大で 213 バイト(型番 10)まで。
    matrix = qr_matrix("https://forms.office.com/r/xxxx")   # True が黒
    png = qr_png("https://...", scale=8)                    # PNG のバイト列
"""
import struct
import zlib

# 型番ごとの (誤り訂正コード語数/ブロック, [(ブロック数, データコード語数), ...])  ※レベル M
EC_M = {
    1: (10, [(1, 16)]), 2: (16, [(1, 28)]), 3: (26, [(1, 44)]), 4: (18, [(2, 32)]), 5: (24, [(2, 43)]),
    6: (16, [(4, 27)]), 7: (18, [(4, 31)]), 8: (22, [(2, 38), (2, 39)]), 9: (22, [(3, 36), (2, 37)]),
    10: (26, [(4, 43), (1, 44)]),
}
ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42],
         9: [6, 26, 46], 10: [6, 28, 50]}

# GF(256)(原始多項式 x^8+x^4+x^3+x^2+1)
EXP = [0] * 512
LOG = [0] * 256
_x = 1
for _i in range(255):
    EXP[_i] = _x
    LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    EXP[_i] = EXP[_i - 255]


def _mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]


def _rs_generator(deg):
    g = [1]
    for i in range(deg):
        ng = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            ng[j] ^= c
            ng[j + 1] ^= _mul(c, EXP[i])
        g = ng
    return g


def _rs_remainder(data, deg):
    g = _rs_generator(deg)
    msg = list(data) + [0] * deg
    for i in range(len(data)):
        c = msg[i]
        if c:
            for j, gj in enumerate(g):
                msg[i + j] ^= _mul(gj, c)
    return msg[len(data):]


def _capacity(v):
    return sum(n * k for n, k in EC_M[v][1])


def _codewords(data, v):
    """データのビット列 → 誤り訂正を付けて並べたコード語"""
    bits = [0, 1, 0, 0]                                   # バイトモード
    nbits = 8 if v <= 9 else 16
    bits += [(len(data) >> (nbits - 1 - i)) & 1 for i in range(nbits)]
    for b in data:
        bits += [(b >> (7 - i)) & 1 for i in range(8)]
    cap = _capacity(v) * 8
    bits += [0] * min(4, cap - len(bits))
    bits += [0] * (-len(bits) % 8)
    cw = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = [0xEC, 0x11]
    while len(cw) < _capacity(v):
        cw.append(pad[(len(cw) - len(bits) // 8) % 2])
    ec, groups = EC_M[v]
    blocks, k = [], 0
    for n, size in groups:
        for _ in range(n):
            blocks.append(cw[k:k + size])
            k += size
    ecb = [_rs_remainder(b, ec) for b in blocks]
    out = []
    for i in range(max(len(b) for b in blocks)):
        out += [b[i] for b in blocks if i < len(b)]
    for i in range(ec):
        out += [b[i] for b in ecb]
    return out


MASKS = [
    lambda x, y: (x + y) % 2 == 0, lambda x, y: y % 2 == 0, lambda x, y: x % 3 == 0, lambda x, y: (x + y) % 3 == 0,
    lambda x, y: (x // 3 + y // 2) % 2 == 0, lambda x, y: x * y % 2 + x * y % 3 == 0,
    lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0, lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0,
]


def _base(v):
    n = 17 + 4 * v
    M = [[False] * n for _ in range(n)]
    F = [[False] * n for _ in range(n)]

    def setf(x, y, dark):
        M[y][x] = dark
        F[y][x] = True

    for i in range(n):                                    # タイミングパターン
        setf(6, i, i % 2 == 0)
        setf(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (n - 4, 3), (3, n - 4)):       # 位置検出パターン + 分離帯
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < n and 0 <= y < n:
                    d = max(abs(dx), abs(dy))
                    setf(x, y, d not in (2, 4))
    pos = ALIGN[v]
    for i, a in enumerate(pos):                           # 位置合わせパターン
        for j, b in enumerate(pos):
            if (i == 0 and j == 0) or (i == 0 and j == len(pos) - 1) or (i == len(pos) - 1 and j == 0):
                continue
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    setf(a + dx, b + dy, max(abs(dx), abs(dy)) != 1)
    _format(M, F, 0, setf, n)                             # 場所を確保(あとで上書き)
    if v >= 7:
        rem = v
        for _ in range(12):
            rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
        bits = v << 12 | rem
        for i in range(18):
            bit = (bits >> i) & 1 == 1
            a, b = n - 11 + i % 3, i // 3
            setf(a, b, bit)
            setf(b, a, bit)
    return M, F, n


def _format(M, F, mask, setf, n):
    data = (0 << 3) | mask                                # レベル M の符号は 00
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    bits = (data << 10 | rem) ^ 0x5412

    def bit(i):
        return (bits >> i) & 1 == 1
    for i in range(6):
        setf(8, i, bit(i))
    setf(8, 7, bit(6))
    setf(8, 8, bit(7))
    setf(7, 8, bit(8))
    for i in range(9, 15):
        setf(14 - i, 8, bit(i))
    for i in range(8):
        setf(n - 1 - i, 8, bit(i))
    for i in range(8, 15):
        setf(8, n - 15 + i, bit(i))
    setf(8, n - 8, True)


def _place(M, F, n, cw):
    i, total = 0, len(cw) * 8
    right = n - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(n):
            for j in range(2):
                x = right - j
                upward = ((right + 1) & 2) == 0
                y = n - 1 - vert if upward else vert
                if not F[y][x]:
                    if i < total:
                        M[y][x] = (cw[i >> 3] >> (7 - (i & 7))) & 1 == 1
                        i += 1
        right -= 2


def _penalty(M, n):
    p = 0
    lines = [M[y] for y in range(n)] + [[M[y][x] for y in range(n)] for x in range(n)]
    for line in lines:
        run, prev = 0, None
        for c in line:
            if c == prev:
                run += 1
            else:
                if run >= 5:
                    p += 3 + run - 5
                run, prev = 1, c
        if run >= 5:
            p += 3 + run - 5
        s = "0000" + "".join("1" if c else "0" for c in line) + "0000"
        for pat in ("10111010000", "00001011101"):
            k = s.find(pat)
            while k != -1:
                p += 40
                k = s.find(pat, k + 1)
    for y in range(n - 1):
        for x in range(n - 1):
            if M[y][x] == M[y][x + 1] == M[y + 1][x] == M[y + 1][x + 1]:
                p += 3
    dark = sum(sum(r) for r in M)
    p += 10 * (abs(dark * 20 - n * n * 10) // (n * n))
    return p


def qr_matrix(text):
    data = text.encode("utf-8")
    for v in range(1, 11):
        head = 4 + (8 if v <= 9 else 16)
        if head + 8 * len(data) <= _capacity(v) * 8:
            break
    else:
        raise ValueError("URL が長すぎる(213 バイトまで)")
    cw = _codewords(data, v)
    best = None
    for mask in range(8):
        M, F, n = _base(v)
        _place(M, F, n, cw)
        for y in range(n):
            for x in range(n):
                if not F[y][x] and MASKS[mask](x, y):
                    M[y][x] = not M[y][x]

        def setf(x, y, dark):
            M[y][x] = dark
        _format(M, F, mask, setf, n)
        p = _penalty(M, n)
        if best is None or p < best[0]:
            best = (p, M)
    return best[1]


def qr_png(text, scale=8, border=4):
    """白黒の PNG(8 ビットグレースケール)"""
    M = qr_matrix(text)
    n = len(M)
    size = (n + 2 * border) * scale
    rows = []
    for yy in range(size):
        y = yy // scale - border
        row = bytearray([0])                              # フィルタなし
        for xx in range(size):
            x = xx // scale - border
            dark = 0 <= x < n and 0 <= y < n and M[y][x]
            row.append(0 if dark else 255)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
