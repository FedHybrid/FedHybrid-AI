import numpy as np

# 가우시안 노이즈 샘플링 함수 추가
def sample_gaussian_noise(size, std=3):
    real = np.random.normal(0, std, size)
    imag = np.random.normal(0, std, size)
    return real + 1j * imag

# 모듈러스 X^4 + 1 적용 함수
def mod_x4_plus_1(poly):
    n = len(poly)
    if n <= 4:
        return poly
    result = np.zeros(4, dtype=np.complex128)
    for i in range(n):
        idx = i % 4
        if (i // 4) % 2 == 1:
            result[idx] -= poly[i]
        else:
            result[idx] += poly[i]
    return result

# 정수 계수로 반올림 함수
def round_to_integer(coeffs):
    return np.array(
        [complex(round(c.real), round(c.imag)) for c in coeffs],
        dtype=np.complex128
    )

# CKKS IDFT 구현 (역 캐노니컬 임베딩) — 1/N 정규화 포함
def ckks_idft(vec, N):
    M = 2 * N
    xi = np.exp(-2j * np.pi / M)
    roots = [xi**(2*k + 1) for k in range(N)]
    result = np.zeros(N, dtype=np.complex128)
    for j in range(N):
        for k in range(N):
            result[j] += vec[k] * np.conj(roots[k])**j
    result /= N
    return result

# 입력 슬롯
z1 = np.array([1+2j, 3+4j], dtype=np.complex128)
z2 = np.array([2+3j, 4+5j], dtype=np.complex128)
Delta = 2**6
print("\n===== 입력 슬롯 =====")
print("z1:", z1)
print("z2:", z2)

# 1) 헤르미티안 확장
z1_h = np.array([z1[0], np.conj(z1[0]), z1[1], np.conj(z1[1])], dtype=np.complex128)
z2_h = np.array([z2[0], np.conj(z2[0]), z2[1], np.conj(z2[1])], dtype=np.complex128)
print("\n===== 헤르미티안 확장 =====")
print("z1_h:", z1_h)
print("z2_h:", z2_h)

# 2) 스케일링
w1 = Delta * z1_h
w2 = Delta * z2_h
print("\n===== 스케일링 (Delta 곱셈) =====")
print("w1:", w1)
print("w2:", w2)

# 3) IDFT → 정수 반올림 → 메시지 다항식 계수
m1 = round_to_integer(ckks_idft(w1, N=4))
m2 = round_to_integer(ckks_idft(w2, N=4))
print("\n===== IDFT 및 메시지 다항식 계수 =====")
print("m1:", m1)
print("m2:", m2)

# 4) 암호화 파라미터
s = np.array([1+0j, 1+0j, 0+0j, 0+0j], dtype=np.complex128)
a = np.array([3+0j, 2+0j, 0+0j, 0+0j], dtype=np.complex128)
e = sample_gaussian_noise(4, std=3)  # 가우시안 노이즈 추가!
print("\n===== 암호화 파라미터 =====")
print("s:", s)
print("a:", a)
print("e (가우시안 노이즈):", e)

# 5) 스위칭키 생성
s2 = mod_x4_plus_1(np.convolve(s, s))
swk_a = a
swk_b = -mod_x4_plus_1(np.convolve(swk_a, s2)) + e
swk = (swk_a, swk_b)
print("\n===== 스위칭키 생성 =====")
print("swk_a:", swk_a)
print("swk_b:", swk_b)

# 6) 암호화 함수
def encrypt(m_coeffs):
    c0 = m_coeffs + mod_x4_plus_1(np.convolve(a, s)) + e
    c1 = -a
    return c0, c1

c0_1, c1_1 = encrypt(m1)
c0_2, c1_2 = encrypt(m2)
print("\n===== 암호화 결과 =====")
print("(c0_1, c1_1):", c0_1, c1_1)
print("(c0_2, c1_2):", c0_2, c1_2)

# 7) 곱셈
d0 = mod_x4_plus_1(np.convolve(c0_1, c0_2))
d1 = mod_x4_plus_1(np.convolve(c0_1, c1_2) + np.convolve(c1_1, c0_2))
d2 = mod_x4_plus_1(np.convolve(c1_1, c1_2))
print("\n===== 곱셈 결과 (d0, d1, d2) =====")
print("d0:", d0)
print("d1:", d1)
print("d2:", d2)

# 8) 리스케일링
d0_ = d0 / Delta
d1_ = d1 / Delta
d2_ = d2 / Delta
print("\n===== 리스케일링 결과 (d0_, d1_, d2_) =====")
print("d0_:", d0_)
print("d1_:", d1_)
print("d2_:", d2_)

# 9) 키 스위칭
e0 = mod_x4_plus_1(np.convolve(d2_, swk[0]))
e1 = mod_x4_plus_1(np.convolve(d2_, swk[1]))
c0_s = d0_ + e0
c1_s = d1_ + e1
print("\n===== 키스위칭 결과 (c0_s, c1_s) =====")
print("c0_s:", c0_s)
print("c1_s:", c1_s)

# 10) 복호화 (Δ 나누지 않음)
dec_poly = c0_s + mod_x4_plus_1(np.convolve(c1_s, s))
print("\n===== 복호화 다항식 (곱셈) =====")
print("dec_poly:", dec_poly)

# 11) 캐노니컬 임베딩 역변환 — 여기만 변경!
N = 4
M = 2 * N
xi = np.exp(-2j * np.pi / M)
roots = [xi**(2*k + 1) for k in range(N)]
poly = np.poly1d(dec_poly[::-1])
decoded = np.array([poly(r) for r in roots], dtype=np.complex128)

# 12) 슬롯 추출 & Δ 역스케일링
recovered = decoded[[0, 2]] / Delta
print("\n===== 복원된 슬롯 (곱셈) =====")
print("recovered:", recovered)
print("평문 곱셈 결과:", z1 * z2)

# CKKS 암호문 덧셈 (리스케일링 없이)
def ckks_add(c1, c2):
    c0_sum = c1[0] + c2[0]
    c1_sum = c1[1] + c2[1]
    return c0_sum, c1_sum

# 예시: m1, m2 암호문 덧셈
c0_add, c1_add = ckks_add((c0_1, c1_1), (c0_2, c1_2))

# 복호화
dec_poly_add = c0_add + mod_x4_plus_1(np.convolve(c1_add, s))
poly_add = np.poly1d(dec_poly_add[::-1])
decoded_add = np.array([poly_add(r) for r in roots], dtype=np.complex128)
recovered_add = decoded_add[[0, 2]] / Delta
print("\n===== 복원된 덧셈 슬롯 =====")
print("recovered_add:", recovered_add)
print("평문 덧셈 결과:", z1 + z2)

def ckks_minus(c1, c2):
    c0_sub = c1[0] - c2[0]
    c1_sub = c1[1] - c2[1]
    return c0_sub, c1_sub

#예시 : m1, m2 암호문 뺄셈
c0_sub, c1_sub = ckks_minus((c0_1, c1_1), (c0_2, c1_2))

dec_poly_sub = c0_sub + mod_x4_plus_1(np.convolve(c1_sub, s))
poly_sub = np.poly1d(dec_poly_sub[::-1])
decoded_sub = np.array([poly_sub(r) for r in roots], dtype=np.complex128)
recovered_sub = decoded_sub[[0, 2]] / Delta
print("\n===== 복원된 뺄셈 슬롯 =====")
print("recovered_sub:", recovered_sub)
print("평문 뺄셈 결과:", z1 - z2)
